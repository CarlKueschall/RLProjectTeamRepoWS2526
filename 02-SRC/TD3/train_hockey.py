"""
AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.
"""

import pickle
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
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
    env = h_env.HockeyEnv(mode=mode, keep_mode=args.keep_mode)
    max_timesteps = get_max_timesteps(mode)
    
    # Verify actual observation dimension matches observation_space
    # (keep_mode may affect observation dimensions: OFF=16, ON=18)
    test_obs, _ = env.reset()
    actual_obs_dim = len(test_obs)
    reported_obs_dim = env.observation_space.shape[0]
    
    if actual_obs_dim != reported_obs_dim:
        print(f"WARNING: Observation dimension mismatch!")
        print(f"  env.observation_space.shape[0] = {reported_obs_dim}")
        print(f"  Actual observation length = {actual_obs_dim}")
        print(f"  keep_mode = {args.keep_mode}")
        print(f"  Using actual observation dimension ({actual_obs_dim}) for agent initialization")
        # Create corrected observation space based on actual observation
        obs_space = spaces.Box(
            low=env.observation_space.low[:actual_obs_dim],
            high=env.observation_space.high[:actual_obs_dim],
            dtype=env.observation_space.dtype
        )
    else:
        obs_space = env.observation_space
    
    print(f"Observation space: {actual_obs_dim} dimensions (keep_mode={args.keep_mode})")

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
                "eval_episodes": args.eval_episodes,
                "self_play_weak_ratio": args.self_play_weak_ratio,
                "use_dual_buffers": args.use_dual_buffers,
                "use_pfsp": args.use_pfsp,
                "pfsp_mode": args.pfsp_mode if args.use_pfsp else None,
                "dynamic_anchor_mixing": args.dynamic_anchor_mixing,
                "performance_gated_selfplay": args.performance_gated_selfplay,
                "selfplay_gate_winrate": args.selfplay_gate_winrate if args.performance_gated_selfplay else None,
                "regression_rollback": args.regression_rollback,
                "regression_threshold": args.regression_threshold if args.regression_rollback else None,
                # New research-based hyperparameters
                "tie_penalty": args.tie_penalty if not args.no_tie_penalty else 0.0,
                "lr_decay": args.lr_decay,
                "lr_min_factor": args.lr_min_factor if args.lr_decay else None,
                "episode_block_size": args.episode_block_size,
                # Strategic rewards configuration (for ablation studies)
                "use_strategic_rewards": args.use_strategic_rewards,
                "strategic_reward_scale": args.strategic_reward_scale if args.use_strategic_rewards else 0.0,
            },
            tags=["TD3", "Hockey", args.mode, args.opponent]
                  + (["self-play"] if args.self_play_start > 0 else [])
                  + (["dual-buffers"] if args.use_dual_buffers else [])
                  + (["PFSP"] if args.use_pfsp else [])
                  + (["tie-penalty"] if not args.no_tie_penalty else [])
                  + (["lr-decay"] if args.lr_decay else [])
                  + (["episode-blocking"] if args.episode_block_size > 1 else [])
                  + (["no-strategic-rewards"] if not args.use_strategic_rewards else [])
                  + (["scaled-strategic"] if args.use_strategic_rewards and args.strategic_reward_scale != 1.0 else [])
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

    #########################################################
    # Load checkpoint first to infer critic_action_dim if needed
    #########################################################
    critic_action_dim = 8  # Default: new version with 8D actions (26D total)
    checkpoint_to_load = None
    i_episode_start = 0
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print("###############################")
        print(f"Analyzing checkpoint: {checkpoint_path}")
        
        checkpoint_to_load = torch.load(checkpoint_path, map_location=device)
        
        # Infer critic_action_dim from checkpoint architecture
        if isinstance(checkpoint_to_load, tuple) and len(checkpoint_to_load) >= 1:
            agent_state = checkpoint_to_load
        elif isinstance(checkpoint_to_load, dict) and 'agent_state' in checkpoint_to_load:
            agent_state = checkpoint_to_load['agent_state']
        else:
            agent_state = checkpoint_to_load

        # Check first critic layer to infer action dimension
        if isinstance(agent_state, tuple) and len(agent_state) >= 1:
            q1_state = agent_state[0]
            if isinstance(q1_state, dict) and 'layers.0.weight' in q1_state:
                critic_input_dim = q1_state['layers.0.weight'].shape[1]
                # Critic input = obs_dim + action_dim
                # If critic_input_dim == 22: old version (18 obs + 4 actions)
                # If critic_input_dim == 26: new version (18 obs + 8 actions)
                if critic_input_dim == 22:
                    critic_action_dim = 4
                    print(f"[CHECKPOINT] Detected OLD format: critic input={critic_input_dim} (18 obs + 4 actions)")
                elif critic_input_dim == 26:
                    critic_action_dim = 8
                    print(f"[CHECKPOINT] Detected NEW format: critic input={critic_input_dim} (18 obs + 8 actions)")
                else:
                    # Try to infer: assume obs_dim=18
                    inferred_action_dim = critic_input_dim - 18
                    if inferred_action_dim in [4, 8]:
                        critic_action_dim = inferred_action_dim
                        print(f"[CHECKPOINT] Inferred action_dim={critic_action_dim} from critic input={critic_input_dim}")
        
        print(f"[CHECKPOINT] Will initialize agent with critic_action_dim={critic_action_dim}")
        print("###############################")

    # create the TD3 agent with all the hyperparams
    agent = TD3Agent(
        obs_space,  # Use corrected observation space if there was a mismatch
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
        critic_action_dim=critic_action_dim,  # Pass inferred dimension
    )

    #########################################################
    # Now load the checkpoint into the agent
    #########################################################
    if checkpoint_to_load is not None:
        print("###############################")
        print(f"Loading checkpoint into agent...")

        if isinstance(checkpoint_to_load, tuple):
            agent.restore_state(checkpoint_to_load)
            print("Loaded TD3 state (Q1, Q2, policy networks)")
        elif isinstance(checkpoint_to_load, dict):
            if 'agent_state' in checkpoint_to_load:
                agent.restore_state(checkpoint_to_load['agent_state'])
                print("Loaded agent state from checkpoint")
                if 'episode' in checkpoint_to_load:
                    i_episode_start = checkpoint_to_load['episode']  # resume from this episode
                    print(f"Resuming from episode {i_episode_start}")

        print("Checkpoint loaded successfully")
        print("###############################")

        checkpoint_name = Path(args.checkpoint).stem
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
    # Initialize LR schedulers (cosine annealing for long training)
    # Research: LR decay improves stability and final performance in 100k+ runs
    #########################################################
    lr_schedulers = None
    if args.lr_decay:
        lr_min_actor = args.lr_actor * args.lr_min_factor
        lr_min_critic = args.lr_critic * args.lr_min_factor

        # Create schedulers for actor and all critic networks
        lr_schedulers = {
            'actor': lr_scheduler.CosineAnnealingLR(
                agent.optimizer, T_max=args.max_episodes, eta_min=lr_min_actor
            ),
            'critic_Q1': lr_scheduler.CosineAnnealingLR(
                agent.Q1.optimizer, T_max=args.max_episodes, eta_min=lr_min_critic
            ),
            'critic_Q2': lr_scheduler.CosineAnnealingLR(
                agent.Q2.optimizer, T_max=args.max_episodes, eta_min=lr_min_critic
            ),
        }
        print(f"LR decay enabled: {args.lr_actor:.0e} -> {lr_min_actor:.0e} over {args.max_episodes} episodes")

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
        regression_rollback=args.regression_rollback,
        regression_threshold=args.regression_threshold,
        observation_space=obs_space,
        action_space=single_player_action_space,
    )
    
    # Set the target episode for self-play activation
    self_play_manager.start_episode = args.self_play_start

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
    print(f"Evaluation interval: {args.eval_interval} episodes")
    print(f"Buffer size: {args.buffer_size:,} transitions")
    # Research-based enhancements
    if not args.no_tie_penalty:
        print(f"Tie penalty: {args.tie_penalty} (encourages decisive wins)")
    if args.lr_decay:
        print(f"LR decay: cosine annealing to {args.lr_min_factor*100:.0f}% of initial")
    if args.episode_block_size > 1:
        print(f"Episode blocking: {args.episode_block_size} episodes per opponent")
    # Strategic rewards configuration (ablation study)
    if args.reward_shaping:
        if not args.use_strategic_rewards:
            print(f"Strategic rewards: DISABLED (PBRS only)")
        elif args.strategic_reward_scale != 1.0:
            print(f"Strategic rewards: SCALED by {args.strategic_reward_scale}x (testing reward hacking)")
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
            print(f"PFSP enabled: {args.pfsp_mode} mode")
        print(f"Episode blocking: {args.episode_block_size} episodes per opponent")
        if args.dynamic_anchor_mixing:
            print("Dynamic anchor mixing enabled (anti-forgetting)")
        if args.performance_gated_selfplay:
            print(f"Performance-gated activation: {args.selfplay_gate_winrate:.0%} win-rate")
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

    #########################################################
    # Episode blocking for opponent selection
    # Research: switching every episode creates unstable Q-targets
    # Play N episodes vs same opponent before switching (AlphaStar-style)
    #########################################################
    current_block_opponent = None  # 'weak', 'strong', or 'self-play'
    block_start_episode = 0

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
        # Manages opponent selection and activation for self-play
        # when self-play infrastructure is initialized
        #########################################################
        opponent_type = None  # 'weak', 'strong', 'self-play', or None (use default)
        if self_play_manager is not None:
            # Check if should activate self-play
            if not self_play_manager.active:
                last_eval_vs_weak = tracker.get_last_eval('weak')

                # Debug logging for activation check (especially around eval episodes)
                if (i_episode % args.eval_interval == 0 or
                    (i_episode > args.self_play_start and i_episode <= args.self_play_start + 5) or
                    (i_episode % 50 == 0 and i_episode >= args.self_play_start)):
                    print(f"\n[ACTIVATION CHECK ep={i_episode}]")
                    print(f"  start_episode={self_play_manager.start_episode}")
                    print(f"  last_eval_vs_weak={last_eval_vs_weak}")
                    print(f"  performance_gated={self_play_manager.performance_gated}")
                    print(f"  gate_winrate={self_play_manager.gate_winrate}")
                    will_activate = self_play_manager.should_activate(i_episode, last_eval_vs_weak)
                    print(f"  → should_activate={will_activate}\n")

                if self_play_manager.should_activate(i_episode, last_eval_vs_weak):
                    self_play_manager.activate(i_episode, selfplay_checkpoints_dir, agent)

            # Update pool with new checkpoint
            removed_episode = self_play_manager.update_pool(i_episode, agent, selfplay_checkpoints_dir)
            if removed_episode:
                print(f"Added ep{i_episode} to pool (removed ep{removed_episode})")

            #########################################################
            # Episode blocking: ONLY apply when self-play is active
            # Research: AlphaStar-style blocking stabilizes Q-learning during self-play
            # Before self-play: use configured opponent (weak/strong) directly
            #########################################################
            if self_play_manager.active:
                # Self-play is active: use episode blocking for opponent selection
                episodes_in_block = i_episode - block_start_episode
                if current_block_opponent is None or episodes_in_block >= args.episode_block_size:
                    # Start new block: select a new opponent type from pool
                    current_block_opponent = self_play_manager.select_opponent(i_episode)
                    block_start_episode = i_episode

                opponent_type = current_block_opponent
            else:
                # Self-play not active yet: use the configured opponent directly
                opponent_type = args.opponent if args.opponent in ['weak', 'strong'] else None
        else:
            # Self-play not configured at all, use the configured opponent
            opponent_type = args.opponent if args.opponent in ['weak', 'strong'] else None

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
            elif opponent_type == 'self-play' and self_play_manager.active and self_play_manager.opponent is not None:
                # Use self-play opponent
                action2 = self_play_manager.get_action(obs_agent2)
                if action2 is None:  # Fallback
                    action2 = weak_eval_bot.act(obs_agent2)
            elif opponent_type == 'weak':
                # Use weak opponent
                action2 = weak_eval_bot.act(obs_agent2)
            elif opponent_type == 'strong':
                # Use strong opponent
                action2 = strong_eval_bot.act(obs_agent2)
            else:
                # Default: use configured opponent (before self-play starts)
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
            # Strategic bonuses (can be disabled to test reward hacking hypothesis)
            #########################################################
            dist_to_puck = np.sqrt((obs_next[0] - obs_next[12])**2 + (obs_next[1] - obs_next[13])**2)  # distance to puck (needed for metrics)
            if args.reward_shaping and args.use_strategic_rewards:
                strategic_bonuses = strategic_shaper.compute(obs_next, info, dist_to_puck)

                # Record opponent position for forcing metric
                strategic_shaper.record_opponent_position([obs_next[6], obs_next[7]])  # track where opponent is

                # Apply strategic bonuses with optional scaling
                for bonus_name, bonus_value in strategic_bonuses.items():
                    r1_shaped += bonus_value * args.strategic_reward_scale  # scaled strategic bonuses
            else:
                strategic_bonuses = {}

            #########################################################
            # Store transition
            #########################################################
            # Store action_combined (8D: 4D agent + 4D opponent) so critic can see full game state
            # This matches the old version where critic had 26D input (18 obs + 8 actions)
            # dual buffers: separate anchor (weak/strong) and pool (self-play) experiences
            if args.use_dual_buffers and hasattr(agent, 'buffer_anchor') and hasattr(agent, 'buffer_pool'):
                # Anchor buffer: weak and strong opponent experiences
                # Pool buffer: self-play experiences
                if opponent_type in ['weak', 'strong'] or (not self_play_manager.active and opponent_type is None):
                    agent.buffer_anchor.add_transition((obs_curr, action_combined.copy(), r1_shaped, obs_next.copy(), float(done or truncated)))
                elif opponent_type == 'self-play':
                    agent.buffer_pool.add_transition((obs_curr, action_combined.copy(), r1_shaped, obs_next.copy(), float(done or truncated)))
                else:
                    # Fallback: use main buffer
                    agent.buffer.add_transition((obs_curr, action_combined.copy(), r1_shaped, obs_next.copy(), float(done or truncated)))
            else:
                agent.buffer.add_transition((obs_curr, action_combined.copy(), r1_shaped, obs_next.copy(), float(done or truncated)))

            # Update tracker
            tracker.add_step_reward(r1_shaped)
            tracker.add_action_magnitude(np.linalg.norm(action1[:2]))  # track action magnitude
            tracker.add_agent_position([obs_next[0], obs_next[1]])  # track agent position
            tracker.add_puck_distance(dist_to_puck)  # track distance to puck

            episode_reward_p1 += r1_shaped
            episode_step_count += 1

            #########################################################
            # Update observations
            #########################################################
            obs = obs_next
            obs_agent2 = env.obs_agent_two()
            # Note: env.obs_agent_two() already returns properly mirrored observations
            # No angle transformation needed - the environment handles this internally

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
        # Step LR schedulers (cosine annealing)
        #########################################################
        if lr_schedulers is not None:
            for scheduler in lr_schedulers.values():
                scheduler.step()

        #########################################################
        # Strategic episode-end bonuses (diversity, forcing)
        #########################################################
        if args.reward_shaping and args.use_strategic_rewards:
            end_bonuses = strategic_shaper.compute_episode_end_bonuses()  # diversity and forcing bonuses
            for bonus_name, bonus_value in end_bonuses.items():
                episode_reward_p1 += bonus_value * args.strategic_reward_scale

        #########################################################
        # Tie penalty (encourages decisive wins over stalemates)
        # Research: ties should be worse than continuing but better than loss
        #########################################################
        tie_penalty_applied = 0.0
        if not args.no_tie_penalty and winner == 0:
            tie_penalty_applied = args.tie_penalty
            episode_reward_p1 += tie_penalty_applied

        #########################################################
        # Update tracker
        #########################################################
        tracker.add_episode_result(episode_reward_p1, episode_step_count, winner)
        tracker.add_strategic_stats(strategic_shaper.get_episode_stats())
        tracker.add_pbrs_total(pbrs_bonus)
        tracker.finalize_episode_behavior_metrics()  # Compute and store behavior metrics for this episode

        #########################################################
        # Self-play result tracking
        #########################################################
        if self_play_manager is not None and self_play_manager.use_pfsp:
            # Record result for PFSP tracking (only for self-play opponents)
            if opponent_type == 'self-play':
                self_play_manager.record_result(winner, False)  # False = not weak opponent
            # Don't record for anchor opponents (weak/strong), only self-play

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
        
        #########################################################
        # Self-play status indicator for progress bar
        #########################################################
        if self_play_manager is not None:
            if self_play_manager.active:
                postfix['SP'] = f'ACTIVE (pool:{len(self_play_manager.pool)})'
            else:
                postfix['SP'] = f'WAITING (ep {args.self_play_start})'
        
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
            # Add behavior metrics
            behavior_metrics = tracker.get_behavior_metrics()
            log_metrics.update(behavior_metrics)

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
            log_metrics["training/tie_penalty"] = tie_penalty_applied

            # LR decay tracking
            if lr_schedulers is not None:
                log_metrics["training/lr_actor"] = lr_schedulers['actor'].get_last_lr()[0]
                log_metrics["training/lr_critic"] = lr_schedulers['critic_Q1'].get_last_lr()[0]

            # Episode blocking tracking (only log when self-play is active)
            if self_play_manager is not None and self_play_manager.active:
                log_metrics["training/episode_block_size"] = args.episode_block_size
                log_metrics["training/episodes_in_current_block"] = i_episode - block_start_episode

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

            #########################################################
            # Self-Play Metrics
            # Detailed insights into opponent mixing, buffer distribution,
            # and PFSP curriculum progression
            # Logged whenever self-play is configured (regardless of activation)
            #########################################################
            if self_play_manager is not None:
                sp_stats = self_play_manager.get_stats()
                
                #########################################################
                # Add stats directly (they already have 'selfplay/' prefix)
                #########################################################
                log_metrics.update(sp_stats)
                
                #########################################################
                # Episode Opponent Type Tracking
                # Logs which opponent type was faced in this episode
                # Useful for understanding opponent mixing during self-play
                #########################################################
                if opponent_type == 'weak':
                    log_metrics['selfplay/episode_opponent_type_weak'] = 1.0
                    log_metrics['selfplay/episode_opponent_type_strong'] = 0.0
                    log_metrics['selfplay/episode_opponent_type_selfplay'] = 0.0
                elif opponent_type == 'strong':
                    log_metrics['selfplay/episode_opponent_type_weak'] = 0.0
                    log_metrics['selfplay/episode_opponent_type_strong'] = 1.0
                    log_metrics['selfplay/episode_opponent_type_selfplay'] = 0.0
                elif opponent_type == 'self-play':
                    log_metrics['selfplay/episode_opponent_type_weak'] = 0.0
                    log_metrics['selfplay/episode_opponent_type_strong'] = 0.0
                    log_metrics['selfplay/episode_opponent_type_selfplay'] = 1.0
                else:
                    # Pre-self-play: training with configured opponent
                    log_metrics['selfplay/episode_opponent_type_weak'] = 0.0
                    log_metrics['selfplay/episode_opponent_type_strong'] = 0.0
                    log_metrics['selfplay/episode_opponent_type_selfplay'] = 0.0

            #########################################################
            # Periodic evaluation (comprehensive three-way evaluation)
            # Always evaluate vs weak, strong, and self-play opponents
            # to track robustness across the full opponent spectrum
            #########################################################
            # Debug: Always check evaluation condition (for debugging)
            eval_modulo = i_episode % args.eval_interval
            should_eval = (eval_modulo == 0)
            if i_episode <= i_episode_start + 30 or should_eval:  # Print for first 30 episodes OR when eval should run
                print(f"[EVAL CHECK ep={i_episode}] eval_interval={args.eval_interval}, modulo={eval_modulo}, should_eval={should_eval}", flush=True)
            
            if should_eval:
                # Debug: Always print when evaluation triggers
                print(f"\n[EVAL TRIGGER] Episode {i_episode}: eval_interval={args.eval_interval}, modulo={i_episode % args.eval_interval}")
                print(f"\n{'='*60}")
                print(f"COMPREHENSIVE EVALUATION at Episode {i_episode}")
                print(f"{'='*60}")
                
                #########################################################
                # Evaluation vs WEAK opponent
                # Baseline: should maintain high win-rate to prevent forgetting
                #########################################################
                print(f"Evaluating vs WEAK opponent (baseline)...")
                eval_weak = evaluate_vs_opponent(
                    agent, weak_eval_bot, mode=mode,
                    num_episodes=args.eval_episodes, max_timesteps=max_timesteps, eval_seed=args.seed, keep_mode=args.keep_mode
                )
                log_metrics["eval/weak/win_rate"] = eval_weak['win_rate']
                log_metrics["eval/weak/win_rate_decisive"] = eval_weak['win_rate_decisive']
                log_metrics["eval/weak/tie_rate"] = eval_weak['tie_rate']
                log_metrics["eval/weak/loss_rate"] = eval_weak['loss_rate']
                log_metrics["eval/weak/avg_reward"] = eval_weak['avg_reward']
                log_metrics["eval/weak/wins"] = eval_weak['wins']
                log_metrics["eval/weak/losses"] = eval_weak['losses']
                log_metrics["eval/weak/ties"] = eval_weak['ties']
                print(f"   ✓ vs WEAK: {eval_weak['win_rate_decisive']:.1%} win (W:{eval_weak['wins']} L:{eval_weak['losses']} T:{eval_weak['ties']}) | Reward: {eval_weak['avg_reward']:.2f}")

                #########################################################
                # Self-play activation check (if configured and not yet active)
                #########################################################
                if self_play_manager is not None and not self_play_manager.active:
                    activation_check = self_play_manager.should_activate(i_episode, eval_weak['win_rate_decisive'])
                    
                    print(f"\n[SELF-PLAY CHECK]")
                    print(f"  Win rate vs weak: {eval_weak['win_rate_decisive']:.1%} (gate: {args.selfplay_gate_winrate:.1%})")
                    print(f"  Status: {'✓ ACTIVATION CONDITIONS MET!' if activation_check else '✗ Waiting for gates...'}")
                    print()

                #########################################################
                # Evaluation vs STRONG opponent
                # Challenge: primary training opponent during pre-self-play phase
                #########################################################
                print(f"Evaluating vs STRONG opponent (training opponent)...")
                eval_strong = evaluate_vs_opponent(
                    agent, strong_eval_bot, mode=mode,
                    num_episodes=args.eval_episodes, max_timesteps=max_timesteps, eval_seed=args.seed, keep_mode=args.keep_mode
                )
                log_metrics["eval/strong/win_rate"] = eval_strong['win_rate']
                log_metrics["eval/strong/win_rate_decisive"] = eval_strong['win_rate_decisive']
                log_metrics["eval/strong/tie_rate"] = eval_strong['tie_rate']
                log_metrics["eval/strong/loss_rate"] = eval_strong['loss_rate']
                log_metrics["eval/strong/avg_reward"] = eval_strong['avg_reward']
                log_metrics["eval/strong/wins"] = eval_strong['wins']
                log_metrics["eval/strong/losses"] = eval_strong['losses']
                log_metrics["eval/strong/ties"] = eval_strong['ties']
                print(f"   ✓ vs STRONG: {eval_strong['win_rate_decisive']:.1%} win (W:{eval_strong['wins']} L:{eval_strong['losses']} T:{eval_strong['ties']}) | Reward: {eval_strong['avg_reward']:.2f}")

                #########################################################
                # Evaluation vs SELF-PLAY opponent (if active)
                # Mirror-match: tests generalization and robustness
                #########################################################
                if self_play_manager.active and self_play_manager.opponent is not None:
                    print(f"Evaluating vs SELF-PLAY opponent (current pool)...")
                    eval_selfplay = evaluate_vs_opponent(
                        agent, self_play_manager.opponent, mode=mode,
                        num_episodes=args.eval_episodes, max_timesteps=max_timesteps, eval_seed=args.seed, keep_mode=args.keep_mode
                    )
                    log_metrics["eval/selfplay/win_rate"] = eval_selfplay['win_rate']
                    log_metrics["eval/selfplay/win_rate_decisive"] = eval_selfplay['win_rate_decisive']
                    log_metrics["eval/selfplay/tie_rate"] = eval_selfplay['tie_rate']
                    log_metrics["eval/selfplay/loss_rate"] = eval_selfplay['loss_rate']
                    log_metrics["eval/selfplay/avg_reward"] = eval_selfplay['avg_reward']
                    log_metrics["eval/selfplay/wins"] = eval_selfplay['wins']
                    log_metrics["eval/selfplay/losses"] = eval_selfplay['losses']
                    log_metrics["eval/selfplay/ties"] = eval_selfplay['ties']
                    print(f"   ✓ vs SELF-PLAY: {eval_selfplay['win_rate_decisive']:.1%} win (W:{eval_selfplay['wins']} L:{eval_selfplay['losses']} T:{eval_selfplay['ties']}) | Reward: {eval_selfplay['avg_reward']:.2f}")
                    log_metrics["eval/selfplay/opponent_age"] = self_play_manager.start_episode - self_play_manager.current_opponent_episode if self_play_manager.active else 0

                print(f"{'='*60}")

                #########################################################
                # Track eval results for regression detection
                #########################################################
                tracker.set_last_eval('weak', eval_weak['win_rate_decisive'])
                tracker.set_peak_eval('weak', max(tracker.get_peak_eval('weak'), eval_weak['win_rate_decisive']))
                tracker.set_last_eval('strong', eval_strong['win_rate_decisive'])

                #########################################################
                # Update self-play manager with eval results (only if self-play is active)
                #########################################################
                if self_play_manager is not None and self_play_manager.active:
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

                #########################################################
                # GIF Recording (comprehensive gameplay visualization)
                # Record behavior during evaluation against all opponent types
                #########################################################
                if not args.no_wandb:
                    try:
                        print(f"Generating gameplay GIFs for episode {i_episode}...")
                        
                        #########################################################
                        # GIF vs WEAK opponent (baseline behavior)
                        #########################################################
                        gif_frames_weak, gif_results_weak = create_gif_for_wandb(
                            env=env, agent=agent, opponent=weak_eval_bot, mode=mode,
                            max_timesteps=max_timesteps, num_episodes=args.gif_episodes,
                            eps=0.0, self_play_opponent=None
                        )
                        save_gif_to_wandb(gif_frames_weak, gif_results_weak, i_episode, run_name,
                                          metric_name="behavior/gameplay_vs_weak")
                        
                        #########################################################
                        # GIF vs STRONG opponent (training opponent)
                        #########################################################
                        gif_frames_strong, gif_results_strong = create_gif_for_wandb(
                            env=env, agent=agent, opponent=strong_eval_bot, mode=mode,
                            max_timesteps=max_timesteps, num_episodes=args.gif_episodes,
                            eps=0.0, self_play_opponent=None
                        )
                        save_gif_to_wandb(gif_frames_strong, gif_results_strong, i_episode, run_name,
                                          metric_name="behavior/gameplay_vs_strong")

                        #########################################################
                        # GIF vs SELF-PLAY opponent (if active and available)
                        # Shows how agent plays against itself / pool opponents
                        #########################################################
                        if self_play_manager.active and self_play_manager.opponent is not None:
                            gif_frames_selfplay, gif_results_selfplay = create_gif_for_wandb(
                                env=env, agent=agent, opponent=self_play_manager.opponent, mode=mode,
                                max_timesteps=max_timesteps, num_episodes=args.gif_episodes,
                                eps=0.0, self_play_opponent=None
                            )
                            save_gif_to_wandb(gif_frames_selfplay, gif_results_selfplay, i_episode, run_name,
                                              metric_name="behavior/gameplay_vs_selfplay")
                        
                        print(f"   ✓ GIFs generated successfully")
                        
                    except Exception as e:
                        print(f"   ✗ GIF recording failed: {e}")

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
                        action_full = batch[0][1]  # 8D action from buffer
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                        action_tensor = torch.FloatTensor(action_full).unsqueeze(0).to(agent.device)  # 8D: [agent, opponent]
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
            if self_play_manager is not None and self_play_manager.regression_rollback:
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
