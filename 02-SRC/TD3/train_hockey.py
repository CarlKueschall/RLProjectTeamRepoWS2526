import pickle
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import wandb
from tqdm import tqdm
from gymnasium import spaces

import hockey.hockey_env as h_env
from hockey.hockey_env import BasicOpponent, Mode

# Import extracted modules
from config import parse_args, get_mode, get_max_timesteps
from agents import TD3Agent, get_device
from evaluation import evaluate_vs_opponent
from visualization import create_gif_for_wandb, save_gif_to_wandb
from metrics import MetricsTracker
from opponents import SelfPlayManager, FixedOpponent
from rewards import PBRSReward, StrategicRewardShaper


def train(args):
    #########################################################
    # Main training loop for hockey
    #########################################################
    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    #########################################################
    # Get device and check features
    #########################################################
    device = get_device(force_cpu=args.cpu)

    print("###############################")
    print(f"Device: {device.type.upper()}")
    print(f"PyTorch: {torch.__version__}")
    print("###############################")

    #########################################################
    # Create environment
    #########################################################
    mode = get_mode(args.mode)
    env = h_env.HockeyEnv(mode=mode, keep_mode=True)
    max_timesteps = get_max_timesteps(mode)

    #########################################################
    # Create opponent
    if args.opponent == 'self':
        opponent = None  # Will create second agent for self-play
    else:
        opponent = FixedOpponent(weak=(args.opponent == 'weak'))

    #########################################################
    # Dedicated bots for evaluation
    #########################################################
    weak_eval_bot = FixedOpponent(weak=True)
    strong_eval_bot = FixedOpponent(weak=False)

    #########################################################
    # Create directories
    #########################################################
    results_dir = Path('./results')
    checkpoints_dir = results_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    selfplay_checkpoints_dir = results_dir / 'selfplay_checkpoints'
    selfplay_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Build run name for wandb
    run_name = f"TD3-Hockey-{args.mode}-{args.opponent}-lr{args.lr_actor:.4f}-seed{args.seed}"

    #########################################################
    # Initialize W&B
    #########################################################
    # logging all the config stuff so we can track experiments
    if not args.no_wandb:
        wandb.init(
            project="rl-hockey",
            name=run_name,
            config={
                "algorithm": "TD3",
                "environment": "Hockey",
                "mode": args.mode,
                "opponent": args.opponent,
                "hidden_sizes_actor": args.hidden_actor,
                "hidden_sizes_critic": args.hidden_critic,
                "learning_rate_actor": args.lr_actor,
                "learning_rate_critic": args.lr_critic,
                "gamma": args.gamma,
                "tau": args.tau,
                "policy_freq": args.policy_freq,
                "target_update_freq": args.target_update_freq,
                "target_noise_std": args.target_noise_std,
                "target_noise_clip": args.target_noise_clip,
                "q_clip": args.q_clip,
                "q_clip_mode": args.q_clip_mode,
                "buffer_size": args.buffer_size,
                "batch_size": args.batch_size,
                "grad_clip": args.grad_clip,
                "train_freq": args.train_freq,
                "grad_accum": args.grad_accum,
                "warmup_episodes": args.warmup_episodes,
                "noise_eps": args.eps,
                "noise_eps_min": args.eps_min,
                "noise_eps_decay": args.eps_decay,
                "device": device.type,
                "random_seed": args.seed,
                "max_episodes": args.max_episodes,
                "self_play_start": args.self_play_start,
                "self_play_pool_size": args.self_play_pool_size,
                "self_play_save_interval": args.self_play_save_interval,
                "eval_interval": args.eval_interval,
                "self_play_weak_ratio": args.self_play_weak_ratio,
                "use_dual_buffers": args.use_dual_buffers,
                "use_pfsp": args.use_pfsp,
                "pfsp_mode": args.pfsp_mode if args.use_pfsp else None,
                "dynamic_anchor_mixing": args.dynamic_anchor_mixing,
                "performance_gated_selfplay": args.performance_gated_selfplay,
                "selfplay_gate_winrate": args.selfplay_gate_winrate if args.performance_gated_selfplay else None,
                "selfplay_gate_variance": args.selfplay_gate_variance if args.performance_gated_selfplay else None,
                "regression_rollback": args.regression_rollback,
                "regression_threshold": args.regression_threshold if args.regression_rollback else None,
            },
            tags=["TD3", "Hockey", args.mode, args.opponent]
                  + (["self-play"] if args.self_play_start > 0 else [])
                  + (["dual-buffers"] if args.use_dual_buffers else [])
                  + (["PFSP"] if args.use_pfsp else [])
        )

    # Track starting episode (for resuming from checkpoint)
    i_episode_start = 0

    #########################################################
    # Initialize agent
    #########################################################
    # Create single-player action space (4 dimensions, not 8)
    single_player_action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(4,),
        dtype=np.float32
    )

    # create the TD3 agent with all the hyperparams
    agent = TD3Agent(
        env.observation_space,
        single_player_action_space,
        eps=args.eps,
        eps_min=args.eps_min,
        eps_decay=args.eps_decay,
        learning_rate_actor=args.lr_actor,
        learning_rate_critic=args.lr_critic,
        discount=args.gamma,
        tau=args.tau,
        policy_freq=args.policy_freq,
        target_update_freq=args.target_update_freq,
        target_noise_std=args.target_noise_std,
        target_noise_clip=args.target_noise_clip,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        hidden_sizes_actor=args.hidden_actor,
        hidden_sizes_critic=args.hidden_critic,
        grad_clip=args.grad_clip,
        force_cpu=args.cpu,
        q_clip=args.q_clip,
        q_clip_mode=args.q_clip_mode,
        use_dual_buffers=args.use_dual_buffers,
    )

    #########################################################
    # Load checkpoint if specified
    #########################################################
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("###############################")
        print(f"Loading checkpoint from: {checkpoint_path}")

        if isinstance(checkpoint, tuple):
            agent.restore_state(checkpoint)
            print("Loaded TD3 state (Q1, Q2, policy networks)")
        elif isinstance(checkpoint, dict):
            if 'agent_state' in checkpoint:
                agent.restore_state(checkpoint['agent_state'])
                print("Loaded agent state from checkpoint")
                if 'episode' in checkpoint:
                    i_episode_start = checkpoint['episode']  # resume from this episode
                    print(f"Resuming from episode {i_episode_start}")

        print("Checkpoint loaded successfully")
        print("###############################")

        checkpoint_name = checkpoint_path.stem
        run_name = f"TD3-TRANSFER-{checkpoint_name}-to-{args.mode}-{args.opponent}"

    #########################################################
    # For self-play, create second agent
    #########################################################
    agent2 = None
    if args.opponent == 'self':  # self-play mode, need two agents
        agent2 = TD3Agent(
            env.observation_space,
            single_player_action_space,
            eps=args.eps,
            eps_min=args.eps_min,
            eps_decay=args.eps_decay,
            learning_rate_actor=args.lr_actor,
            learning_rate_critic=args.lr_critic,
            discount=args.gamma,
            tau=args.tau,
            policy_freq=args.policy_freq,
            target_update_freq=args.target_update_freq,
            target_noise_std=args.target_noise_std,
            target_noise_clip=args.target_noise_clip,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            hidden_sizes_actor=args.hidden_actor,
            hidden_sizes_critic=args.hidden_critic,
            grad_clip=args.grad_clip,
            force_cpu=args.cpu,
            q_clip=args.q_clip,
            q_clip_mode=args.q_clip_mode,
        )
        torch.manual_seed(args.seed + 1)  # different seed for agent2
        np.random.seed(args.seed + 1)

    #########################################################
    # Initialize reward shapers
    #########################################################
    pbrs_shaper = PBRSReward(gamma=args.gamma, annealing_episodes=5000)
    if args.reward_shaping and args.self_play_start > 0:
        pbrs_shaper.set_self_play_start(args.self_play_start)  # anneal during self-play

    strategic_shaper = StrategicRewardShaper()

    #########################################################
    # Initialize self-play manager
    #########################################################
    self_play_manager = SelfPlayManager(
        pool_size=args.self_play_pool_size,
        save_interval=args.self_play_save_interval,
        weak_ratio=args.self_play_weak_ratio,
        device=device,
        use_pfsp=args.use_pfsp,
        pfsp_mode=args.pfsp_mode,
        dynamic_anchor_mixing=args.dynamic_anchor_mixing,
        performance_gated=args.performance_gated_selfplay,
        gate_winrate=args.selfplay_gate_winrate,
        gate_variance=args.selfplay_gate_variance,
        regression_rollback=args.regression_rollback,
        regression_threshold=args.regression_threshold,
    )

    #########################################################
    # Initialize metrics tracker
    #########################################################
    tracker = MetricsTracker(rolling_window=100)  # track wins/losses/rewards etc

    print("###############################")
    print(f"Training TD3 on Hockey ({args.mode} mode)")
    print(f"Run: {run_name}")
    print(f"Opponent: {args.opponent}")
    print(f"Max timesteps per episode: {max_timesteps}")
    print(f"Warmup episodes: {args.warmup_episodes} (NO training, just exploration)")
    print(f"Epsilon: {args.eps} -> {args.eps_min} (decay: {args.eps_decay})")
    print("###############################")

    if args.self_play_start > 0:
        print("###############################")
        print("SELF-PLAY ENABLED")
        print(f"Starts at episode: {args.self_play_start}")
        print(f"Pool size: {args.self_play_pool_size}")
        print(f"Save interval: {args.self_play_save_interval}")
        print(f"Eval interval: {args.eval_interval}")
        print(f"Weak opponent ratio: {args.self_play_weak_ratio:.0%}")
        if args.use_pfsp:
            print(f"PFSP enabled: {args.pfsp_mode} curriculum")
        if args.dynamic_anchor_mixing:
            print("Dynamic anchor mixing enabled (anti-forgetting)")
        if args.performance_gated_selfplay:
            print(f"Performance-gated activation: {args.selfplay_gate_winrate:.0%} win-rate, {args.selfplay_gate_variance:.2f} variance")
        if args.regression_rollback:
            print(f"Regression rollback enabled: threshold {args.regression_threshold:.0%}")
        print("###############################")

    #########################################################
    # Training loop
    #########################################################
    global_step = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_episode = 0

    pbar = tqdm(range(i_episode_start + 1, args.max_episodes + 1), desc="Training", unit="ep")

    for i_episode in pbar:
        #########################################################
        # Use random seeds for state diversity
        #########################################################
        if args.seed is not None:
            np.random.seed(args.seed + i_episode)
            reset_seed = np.random.randint(0, 1000000)
        else:
            reset_seed = None
        obs, info = env.reset(seed=reset_seed)
        obs_agent2 = env.obs_agent_two()
        agent.reset()
        if agent2:
            agent2.reset()

        #########################################################
        # Reset shapers and trackers for new episode
        #########################################################
        strategic_shaper.reset()
        pbrs_shaper.reset()
        tracker.reset_episode()

        #########################################################
        # Self-play management
        #########################################################
        if args.self_play_start > 0:
            # Check if should activate self-play
            if not self_play_manager.active:
                rolling_variance = np.std(list(tracker.rolling_outcomes)) if len(tracker.rolling_outcomes) > 0 else 0.0
                last_eval_vs_weak = tracker.get_last_eval('weak')
                if self_play_manager.should_activate(i_episode, last_eval_vs_weak, rolling_variance):
                    self_play_manager.activate(i_episode, selfplay_checkpoints_dir, agent)

            # Update pool with new checkpoint
            removed_episode = self_play_manager.update_pool(i_episode, agent, selfplay_checkpoints_dir)
            if removed_episode:
                print(f"Added ep{i_episode} to pool (removed ep{removed_episode})")

            # Select opponent for this episode
            use_weak_this_episode = self_play_manager.select_opponent(i_episode)
        else:
            use_weak_this_episode = False

        #########################################################
        # Episode loop
        #########################################################
        episode_reward_p1 = 0
        episode_step_count = 0
        winner = 0  # Initialize winner (default to tie)

        for t in range(max_timesteps):
            global_step += 1
            obs_curr = obs.copy()

            #########################################################
            # Agent 1 action
            #########################################################
            action1 = agent.act(obs)

            #########################################################
            # Agent 2 action
            #########################################################
            if args.opponent == 'self':
                action2 = agent2.act(obs_agent2)
            elif self_play_manager.active and self_play_manager.opponent is not None and not use_weak_this_episode:
                action2 = self_play_manager.get_action(obs_agent2)
                if action2 is None:  # Fallback
                    action2 = opponent.act(obs_agent2)
            else:
                action2 = opponent.act(obs_agent2)

            #########################################################
            # Action slicing (critical fix)
            #########################################################
            # primary agent might have 8 outputs, but we only want its P1 actions
            # By slicing both to [:4], we ensure proper player separation
            action_combined = np.hstack([action1[:4], action2[:4]])

            # Step environment
            obs_next, r1, done, truncated, info = env.step(action_combined)

            #########################################################
            # Apply reward shaping
            #########################################################
            if args.reward_shaping:
                pbrs_bonus = pbrs_shaper.compute(obs_curr, obs_next, done=(done or truncated), episode=i_episode)
                r1_shaped = r1 + pbrs_bonus  # add PBRS bonus to sparse reward
            else:
                r1_shaped = r1
                pbrs_bonus = 0.0

            #########################################################
            # Strategic bonuses
            #########################################################
            dist_to_puck = np.sqrt((obs_next[0] - obs_next[12])**2 + (obs_next[1] - obs_next[13])**2)  # distance to puck
            strategic_bonuses = strategic_shaper.compute(obs_next, info, dist_to_puck)

            # Record opponent position for forcing metric
            if args.reward_shaping:
                strategic_shaper.record_opponent_position([obs_next[6], obs_next[7]])  # track where opponent is

            # Apply strategic bonuses
            for bonus_name, bonus_value in strategic_bonuses.items():
                r1_shaped += bonus_value  # add all the strategic bonuses

            #########################################################
            # Store transition
            #########################################################
            # dual buffers: separate anchor (weak) and pool (self-play) experiences
            if args.use_dual_buffers and hasattr(agent, 'buffer_anchor') and hasattr(agent, 'buffer_pool'):
                if use_weak_this_episode or not self_play_manager.active:
                    agent.buffer_anchor.add_transition((obs_curr, action1, r1_shaped, obs_next.copy(), float(done or truncated)))
                else:
                    agent.buffer_pool.add_transition((obs_curr, action1, r1_shaped, obs_next.copy(), float(done or truncated)))
            else:
                agent.buffer.add_transition((obs_curr, action1, r1_shaped, obs_next.copy(), float(done or truncated)))

            # Update tracker
            tracker.add_step_reward(r1_shaped)
            tracker.add_action_magnitude(np.linalg.norm(action1[:2]))  # track action magnitude

            episode_reward_p1 += r1_shaped
            episode_step_count += 1

            #########################################################
            # Update observations
            #########################################################
            obs = obs_next
            obs_agent2 = env.obs_agent_two()
            # FIX: Mirror angles for P2
            obs_agent2[2] = np.arctan2(-np.sin(obs_agent2[2]), -np.cos(obs_agent2[2]))
            obs_agent2[8] = np.arctan2(-np.sin(obs_agent2[8]), -np.cos(obs_agent2[8]))

            #########################################################
            # Train during episode
            #########################################################
            warmup_complete = i_episode >= args.warmup_episodes  # wait for warmup before training
            if args.train_freq != -1 and len(agent.buffer) >= args.batch_size and warmup_complete:
                if global_step % args.train_freq == 0:
                    scaled_iterations = max(1, int(32 / (250 / args.train_freq)))  # scale training iterations based on freq
                    losses = agent.train(iter_fit=scaled_iterations)
                    if losses:
                        tracker.add_losses(losses)

            if done or truncated:
                winner = info.get('winner', 0)
                if winner not in [-1, 0, 1]:
                    winner = 0
                break

        #########################################################
        # Train after episode (if train_freq == -1)
        #########################################################
        if args.train_freq == -1 and warmup_complete:
            if len(agent.buffer) >= args.batch_size:
                losses = agent.train(iter_fit=32)
                if losses:
                    tracker.add_losses(losses)

        #########################################################
        # Train agent2 in self-play
        #########################################################
        if agent2 and len(agent2.buffer) >= args.batch_size:
            agent2.train(iter_fit=32)  # train the second agent too

        #########################################################
        # Decay exploration
        #########################################################
        agent.decay_epsilon()
        if agent2:
            agent2.decay_epsilon()

        #########################################################
        # Strategic episode-end bonuses
        #########################################################
        end_bonuses = strategic_shaper.compute_episode_end_bonuses()  # diversity and forcing bonuses
        for bonus_name, bonus_value in end_bonuses.items():
            episode_reward_p1 += bonus_value

        #########################################################
        # Update tracker
        #########################################################
        tracker.add_episode_result(episode_reward_p1, episode_step_count, winner)
        tracker.add_strategic_stats(strategic_shaper.get_episode_stats())
        tracker.add_pbrs_total(pbrs_bonus)

        #########################################################
        # Self-play result tracking
        #########################################################
        if args.self_play_start > 0 and self_play_manager.use_pfsp:
            self_play_manager.record_result(winner, use_weak_this_episode)

        #########################################################
        # Update progress bar
        #########################################################
        win_rate = tracker.get_win_rate()
        postfix = {
            'reward': f'{episode_reward_p1:6.1f}',
            'avg': f'{tracker.get_avg_reward():6.1f}',
            'win_rate': f'{win_rate:.2%}',
            'wins': f'{tracker.wins}/{tracker.total_games}'
        }
        if self_play_manager.active:
            postfix['mode'] = 'SELF-PLAY'
        pbar.set_postfix(postfix)

        #########################################################
        # Logging
        #########################################################
        if i_episode % args.log_interval == 0:
            current_time = time.time()
            episodes_since_last_log = i_episode - last_log_episode
            time_since_last_log = current_time - last_log_time
            eps_per_sec = episodes_since_last_log / time_since_last_log if time_since_last_log > 0 else 0

            #########################################################
            # Build log metrics
            #########################################################
            log_metrics = tracker.get_log_metrics()
            log_metrics["performance/cumulative_win_rate"] = win_rate
            log_metrics["performance/wins"] = tracker.wins
            log_metrics["performance/losses"] = tracker.losses
            log_metrics["performance/ties"] = tracker.ties
            log_metrics["scoring/goals_scored"] = tracker.goals_scored
            log_metrics["scoring/goals_conceded"] = tracker.goals_conceded
            log_metrics["training/epsilon"] = agent.get_epsilon()
            log_metrics["training/eps_per_sec"] = eps_per_sec
            log_metrics["training/episode"] = i_episode
            log_metrics["training/pbrs_enabled"] = args.reward_shaping

            if args.reward_shaping:
                log_metrics["pbrs/avg_per_episode"] = tracker.get_avg_pbrs()
                if self_play_manager.active:
                    log_metrics["pbrs/annealing_weight"] = pbrs_shaper.get_annealing_weight(i_episode)

                #########################################################
                # Strategic reward shaping metrics
                #########################################################
                if tracker.strategic_stats:
                    log_metrics["strategic/shots_clear"] = tracker.strategic_stats.get('shots_clear', 0)
                    log_metrics["strategic/shots_blocked"] = tracker.strategic_stats.get('shots_blocked', 0)
                    log_metrics["strategic/shot_quality_ratio"] = tracker.strategic_stats.get('shot_quality_ratio', 0.0)
                    log_metrics["strategic/attack_sides_unique"] = tracker.strategic_stats.get('attack_sides_unique', 0)
                    log_metrics["strategic/attack_diversity_bonus"] = tracker.strategic_stats.get('attack_diversity_bonus', 0.0)
                    log_metrics["strategic/opponent_total_movement"] = tracker.strategic_stats.get('total_opponent_movement', 0.0)
                    log_metrics["strategic/opponent_avg_movement"] = tracker.strategic_stats.get('avg_opponent_movement', 0.0)
                    log_metrics["strategic/forcing_bonus"] = tracker.strategic_stats.get('forcing_bonus', 0.0)

            if args.self_play_start > 0:
                log_metrics.update(self_play_manager.get_stats())

            #########################################################
            # Periodic evaluation (always runs against both weak and strong)
            #########################################################
            if i_episode % args.eval_interval == 0:
                print(f"\nEvaluating vs WEAK opponent...")
                eval_weak = evaluate_vs_opponent(
                    agent, weak_eval_bot, mode=mode,
                    num_episodes=100, max_timesteps=max_timesteps, eval_seed=args.seed
                )
                log_metrics["eval/vs_weak_win_rate"] = eval_weak['win_rate']
                log_metrics["eval/vs_weak_win_rate_decisive"] = eval_weak['win_rate_decisive']
                log_metrics["eval/vs_weak_tie_rate"] = eval_weak['tie_rate']
                log_metrics["eval/vs_weak_loss_rate"] = eval_weak['loss_rate']
                log_metrics["eval/vs_weak_avg_reward"] = eval_weak['avg_reward']
                print(f"   [EVAL] vs weak: {eval_weak['win_rate']:.1%} W/L/T")

                # Also evaluate vs strong
                print(f"Evaluating vs STRONG opponent...")
                eval_strong = evaluate_vs_opponent(
                    agent, strong_eval_bot, mode=mode,
                    num_episodes=100, max_timesteps=max_timesteps, eval_seed=args.seed
                )
                log_metrics["eval/vs_strong_win_rate"] = eval_strong['win_rate']
                log_metrics["eval/vs_strong_win_rate_decisive"] = eval_strong['win_rate_decisive']
                log_metrics["eval/vs_strong_tie_rate"] = eval_strong['tie_rate']
                log_metrics["eval/vs_strong_loss_rate"] = eval_strong['loss_rate']
                log_metrics["eval/vs_strong_avg_reward"] = eval_strong['avg_reward']
                print(f"   [EVAL] vs strong: {eval_strong['win_rate']:.1%} W/L/T")

                #########################################################
                # Track eval results
                #########################################################
                tracker.set_last_eval('weak', eval_weak['win_rate_decisive'])
                tracker.set_peak_eval('weak', max(tracker.get_peak_eval('weak'), eval_weak['win_rate_decisive']))
                tracker.set_last_eval('strong', eval_strong['win_rate_decisive'])

                #########################################################
                # Update self-play manager with eval results (if self-play enabled)
                #########################################################
                if args.self_play_start > 0:
                    if self_play_manager.dynamic_anchor_mixing:
                        last_eval = tracker.get_last_eval('weak')
                        peak_eval = tracker.get_peak_eval('weak')
                        drop_from_peak = peak_eval - eval_weak['win_rate_decisive'] if peak_eval > 0 else 0.0
                        self_play_manager.update_anchor_ratio(drop_from_peak)

                    if self_play_manager.regression_rollback:
                        should_rollback, rollback_path = self_play_manager.check_regression(eval_weak['win_rate_decisive'])
                        if should_rollback:
                            print("###############################")
                            print("REGRESSION ROLLBACK TRIGGERED")
                            print(f"Rolling back to: {rollback_path}")
                            print("###############################")
                            if rollback_path and Path(rollback_path).exists():
                                checkpoint = torch.load(rollback_path, map_location=device)
                                agent.restore_state(checkpoint['agent_state'])
                            else:
                                print(f"WARNING: Rollback path not found or invalid: {rollback_path}")

                print("")

            #########################################################
            # Q-value monitoring
            #########################################################
            # check q-values to see if they're exploding or something
            if i_episode % 100 == 0 and len(agent.buffer) >= args.batch_size:
                q_values = []
                with torch.no_grad():
                    for _ in range(min(100, len(agent.buffer))):
                        batch = agent.buffer.sample(1)
                        state = batch[0][0]
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                        action_tensor = agent.policy(state_tensor)
                        q_value = agent.Q1.Q_value(state_tensor, action_tensor).item()
                        q_values.append(q_value)

                if q_values:
                    log_metrics["values/Q_avg"] = np.mean(q_values)
                    log_metrics["values/Q_std"] = np.std(q_values)
                    log_metrics["values/Q_min"] = np.min(q_values)
                    log_metrics["values/Q_max"] = np.max(q_values)

            #########################################################
            # Progress update
            #########################################################
            if args.reward_shaping:
                tqdm.write(f'Episode {i_episode:4d} | '
                           f'P1 Reward: {episode_reward_p1:7.2f} | '
                           f'Win Rate: {win_rate:5.1%} | '
                           f'PBRS: {tracker.get_avg_pbrs():6.2f} | '
                           f'{eps_per_sec:.2f} eps/s')
            else:
                tqdm.write(f'Episode {i_episode:4d} | '
                           f'P1 Reward: {episode_reward_p1:7.2f} | '
                           f'Win Rate: {win_rate:5.1%} | '
                           f'{eps_per_sec:.2f} eps/s')

            if not args.no_wandb:
                wandb.log(log_metrics, step=i_episode)

            last_log_time = current_time
            last_log_episode = i_episode

        #########################################################
        # GIF Recording
        #########################################################
        if args.gif_interval > 0 and (i_episode == 1 or i_episode % args.gif_interval == 0) and not args.no_wandb:
            try:
                if self_play_manager.active and self_play_manager.opponent is not None:
                    # Record vs target opponent
                    gif_frames_target, gif_results_target = create_gif_for_wandb(
                        env=env, agent=agent, opponent=opponent, mode=mode,
                        max_timesteps=max_timesteps, num_episodes=args.gif_episodes,
                        eps=0.0, self_play_opponent=None
                    )
                    save_gif_to_wandb(gif_frames_target, gif_results_target, i_episode, run_name,
                                      metric_name="behavior/gameplay_gif_vs_target")

                    # Record vs self-play opponent
                    gif_frames_selfplay, gif_results_selfplay = create_gif_for_wandb(
                        env=env, agent=agent, opponent=opponent, mode=mode,
                        max_timesteps=max_timesteps, num_episodes=args.gif_episodes,
                        eps=0.0, self_play_opponent=self_play_manager.opponent
                    )
                    save_gif_to_wandb(gif_frames_selfplay, gif_results_selfplay, i_episode, run_name,
                                      metric_name="behavior/gameplay_gif_vs_selfplay")
                else:
                    gif_frames, gif_results = create_gif_for_wandb(
                        env=env, agent=agent, opponent=opponent, mode=mode,
                        max_timesteps=max_timesteps, num_episodes=args.gif_episodes,
                        eps=0.0, self_play_opponent=None
                    )
                    save_gif_to_wandb(gif_frames, gif_results, i_episode, run_name)
            except Exception as e:
                print(f"GIF recording failed at episode {i_episode}: {e}")

        #########################################################
        # Save checkpoint
        #########################################################
        if i_episode % args.save_interval == 0:
            checkpoint_path = checkpoints_dir / f'TD3_Hockey_{args.mode}_{args.opponent}_{i_episode}_seed{args.seed}.pth'
            checkpoint_data = {
                'agent_state': agent.state(),
                'episode': i_episode,
            }
            torch.save(checkpoint_data, checkpoint_path)
            if not args.no_wandb:
                wandb.save(str(checkpoint_path))
            print(f'--- Checkpoint saved: {checkpoint_path.name} ---')
            
            # Update best checkpoint for regression rollback if enabled
            if args.self_play_start > 0 and self_play_manager.regression_rollback:
                # Check if this checkpoint corresponds to the best eval we've seen
                # (check_regression updates best_eval_vs_weak when it finds a new best)
                last_eval = tracker.get_last_eval('weak')
                if last_eval is not None and abs(last_eval - self_play_manager.best_eval_vs_weak) < 0.001:
                    # This checkpoint corresponds to the best eval (within small tolerance for float comparison)
                    self_play_manager.set_best_checkpoint(str(checkpoint_path))

    #########################################################
    # Final summary
    #########################################################
    training_time = time.time() - start_time
    final_reward = np.mean(tracker.rewards_p1[-20:]) if len(tracker.rewards_p1) >= 20 else 0
    final_win_rate = tracker.get_win_rate()
    final_eps_per_sec = args.max_episodes / training_time if training_time > 0 else 0

    print("###############################")
    print("Training Complete!")
    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Final reward (last 20): {final_reward:.2f}")
    print(f"Final win rate: {final_win_rate:.1%}")
    print(f"Wins: {tracker.wins}, Losses: {tracker.losses}, Ties: {tracker.ties}")
    print(f"Goals scored: {tracker.goals_scored}, Conceded: {tracker.goals_conceded}")
    print(f"Average speed: {final_eps_per_sec:.2f} eps/s ({training_time/args.max_episodes:.2f} s/ep)")
    print("###############################")

    #########################################################
    # Save final statistics
    #########################################################
    stats_path = results_dir / f'TD3_Hockey_{args.mode}_{args.opponent}_stats_seed{args.seed}.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump({
            "rewards_p1": tracker.rewards_p1,
            "wins": tracker.wins,
            "losses": tracker.losses,
            "ties": tracker.ties,
            "goals_scored": tracker.goals_scored,
            "goals_conceded": tracker.goals_conceded,
            "config": vars(args),
        }, f)

    # Save final model
    final_model_path = checkpoints_dir / f'TD3_Hockey_{args.mode}_{args.opponent}_final_seed{args.seed}.pth'
    torch.save(agent.state(), final_model_path)

    #########################################################
    # Log final summary to W&B
    #########################################################
    if not args.no_wandb:
        wandb.summary["final_reward"] = final_reward
        wandb.summary["final_win_rate"] = final_win_rate
        wandb.summary["wins"] = tracker.wins
        wandb.summary["losses"] = tracker.losses
        wandb.summary["ties"] = tracker.ties
        wandb.summary["goals_scored"] = tracker.goals_scored
        wandb.summary["goals_conceded"] = tracker.goals_conceded
        wandb.summary["training_time_seconds"] = training_time
        wandb.summary["total_episodes"] = args.max_episodes
        wandb.summary["eps_per_sec"] = final_eps_per_sec
        wandb.save(str(stats_path))
        wandb.save(str(final_model_path))
        wandb.finish()

    env.close()

    return agent, tracker.rewards_p1


if __name__ == '__main__':
    args = parse_args()
    train(args)
