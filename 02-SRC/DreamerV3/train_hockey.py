"""
DreamerV3 Training Script for Hockey.

AI Usage Declaration:
This file was developed with assistance from Claude Code.

Based on NaturalDreamer, adapted for hockey with:
- 18-dim vector observations (MLP encoder/decoder)
- Auxiliary tasks for world model representation learning
- Opponent management (weak/strong/self-play)
- W&B logging and metrics tracking
- GIF recording for visualization
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import wandb
import hockey.hockey_env as h_env
from hockey.hockey_env import Mode

from dreamer import Dreamer
from utils import loadConfig, seedEverything, ensureParentFolders, saveLossesToCSV, plotMetrics
from opponents import FixedOpponent, SelfPlayManager


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_opponent(opponent_type: str):
    """Create opponent based on type."""
    if opponent_type == "weak":
        return FixedOpponent(weak=True)
    elif opponent_type == "strong":
        return FixedOpponent(weak=False)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")


def run_episode(env, agent, opponent, training=True, seed=None, render=False):
    """
    Run single episode, collecting transitions.

    Returns:
        total_reward: Sparse game reward
        steps: Number of steps
        outcome: 'win', 'loss', or 'draw'
        transitions: List of (obs, action, reward, next_obs, done) tuples
        frames: List of RGB frames if render=True
    """
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    h, z = None, None  # Recurrent states
    total_reward = 0.0
    steps = 0
    transitions = []
    frames = []
    done = False

    while not done:
        # Capture frame if rendering
        if render:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)

        # Get agent action
        action, h, z = agent.act(obs, h, z)

        # Get opponent action
        obs_opponent = env.obs_agent_two()
        action_opponent = opponent.act(obs_opponent)

        # Step environment
        next_obs, reward, done, truncated, info = env.step(np.hstack([action, action_opponent]))
        done = done or truncated

        # Store transition for buffer
        if training:
            transitions.append((obs, action, reward, next_obs, done))

        total_reward += reward
        obs = next_obs
        steps += 1

    # Determine outcome
    if info.get('winner', 0) == 1:
        outcome = 'win'
    elif info.get('winner', 0) == -1:
        outcome = 'loss'
    else:
        outcome = 'draw'

    return total_reward, steps, outcome, transitions, frames


def record_gif(env, agent, opponent, num_episodes=3, max_timesteps=250):
    """
    Record GIF of evaluation episodes.

    Returns:
        frames: List of stitched RGB frames
        results: List of outcomes
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed. GIF recording disabled.")
        return None, []

    all_episode_frames = []
    results = []

    for _ in range(num_episodes):
        _, _, outcome, _, frames = run_episode(
            env, agent, opponent, training=False, render=True
        )
        all_episode_frames.append(frames)
        results.append(1 if outcome == 'win' else (-1 if outcome == 'loss' else 0))

    if not all_episode_frames or not all_episode_frames[0]:
        return None, results

    # Find max frames and pad shorter episodes
    max_frames = max(len(f) for f in all_episode_frames)
    for frames in all_episode_frames:
        if len(frames) < max_frames and len(frames) > 0:
            while len(frames) < max_frames:
                frames.append(frames[-1])

    # Stitch frames horizontally
    stitched_frames = []
    for frame_idx in range(max_frames):
        episode_frames = [ep[frame_idx] for ep in all_episode_frames if frame_idx < len(ep)]
        if episode_frames:
            pil_images = [Image.fromarray(f) for f in episode_frames]
            total_width = sum(img.width for img in pil_images)
            max_height = max(img.height for img in pil_images)
            stitched = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in pil_images:
                stitched.paste(img, (x_offset, 0))
                x_offset += img.width
            stitched_frames.append(np.array(stitched))

    return stitched_frames, results


def save_gif_to_wandb(frames, results, step, opponent_name):
    """Save GIF to W&B."""
    if frames is None or len(frames) == 0:
        return

    try:
        import imageio
        import tempfile
    except ImportError:
        return

    try:
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            tmp_path = tmp.name

        imageio.mimsave(tmp_path, frames, fps=15, loop=0)

        result_strs = []
        for i, r in enumerate(results):
            if r == 1:
                result_strs.append(f"Ep{i+1}:WIN")
            elif r == -1:
                result_strs.append(f"Ep{i+1}:LOSS")
            else:
                result_strs.append(f"Ep{i+1}:DRAW")

        wins = sum(1 for r in results if r == 1)
        caption = f"Step {step:,} vs {opponent_name} | {' | '.join(result_strs)} | {wins}/{len(results)} wins"

        wandb.log({
            f"eval/gif_{opponent_name}": wandb.Video(tmp_path, fps=15, format="gif", caption=caption),
        }, step=step)

        os.unlink(tmp_path)
        print(f"  GIF recorded vs {opponent_name}: {', '.join(result_strs)}")

    except Exception as e:
        print(f"GIF recording failed: {e}")


def evaluate_against_opponent(env, agent, opponent, num_episodes, opponent_name):
    """
    Evaluate agent against a specific opponent.

    Args:
        env: Hockey environment
        agent: DreamerV3 agent
        opponent: Opponent to evaluate against
        num_episodes: Number of episodes to run
        opponent_name: Name for logging

    Returns:
        dict: Evaluation metrics
    """
    eval_rewards = []
    eval_outcomes = {'win': 0, 'loss': 0, 'draw': 0}

    for _ in range(num_episodes):
        opponent.reset()  # Reset opponent state if any
        reward, _, outcome, _, _ = run_episode(env, agent, opponent, training=False)
        eval_rewards.append(reward)
        eval_outcomes[outcome] += 1

    win_rate = eval_outcomes['win'] / num_episodes
    mean_reward = np.mean(eval_rewards)

    return {
        f'eval/{opponent_name}_win_rate': win_rate,
        f'eval/{opponent_name}_mean_reward': mean_reward,
        f'eval/{opponent_name}_wins': eval_outcomes['win'],
        f'eval/{opponent_name}_losses': eval_outcomes['loss'],
        f'eval/{opponent_name}_draws': eval_outcomes['draw'],
    }


def record_and_log_gif(env, agent, opponent, opponent_name, step, num_episodes, use_wandb):
    """
    Record GIF against opponent and log to W&B.

    Args:
        env: Hockey environment
        agent: DreamerV3 agent
        opponent: Opponent for GIF
        opponent_name: Name for logging
        step: Current gradient step
        num_episodes: Episodes per GIF
        use_wandb: Whether W&B is enabled
    """
    if not use_wandb:
        return

    opponent.reset()
    frames, results = record_gif(env, agent, opponent, num_episodes=num_episodes)
    save_gif_to_wandb(frames, results, step, opponent_name)


def parse_args():
    """Parse command line arguments with all config overrides."""
    parser = argparse.ArgumentParser(description="DreamerV3 Hockey Training")

    # Config file
    parser.add_argument("--config", type=str, default="hockey.yml", help="Config file")

    # Basic settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for logging")

    # Environment
    parser.add_argument("--opponent", type=str, default=None, choices=["weak", "strong"],
                        help="Opponent type")
    parser.add_argument("--mode", type=str, default="NORMAL",
                        choices=["NORMAL", "TRAIN_SHOOTING", "TRAIN_DEFENSE"],
                        help="Environment mode")

    # Training schedule
    parser.add_argument("--gradient_steps", type=int, default=None, help="Total gradient steps")
    parser.add_argument("--replay_ratio", type=int, default=None,
                        help="Gradient steps per environment step")
    parser.add_argument("--warmup_episodes", type=int, default=None, help="Warmup episodes")
    parser.add_argument("--interaction_episodes", type=int, default=None,
                        help="Episodes between training batches")

    # DreamerV3 hyperparameters
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--batch_length", type=int, default=None, help="Sequence length")
    parser.add_argument("--imagination_horizon", type=int, default=None, help="Imagination horizon")
    parser.add_argument("--recurrent_size", type=int, default=None, help="GRU hidden size")
    parser.add_argument("--latent_length", type=int, default=None, help="Number of latent vars")
    parser.add_argument("--latent_classes", type=int, default=None, help="Classes per latent var")
    parser.add_argument("--encoded_obs_size", type=int, default=None, help="Encoded observation size")

    # Learning rates
    parser.add_argument("--lr_world", type=float, default=None, help="World model LR")
    parser.add_argument("--lr_actor", type=float, default=None, help="Actor LR")
    parser.add_argument("--lr_critic", type=float, default=None, help="Critic LR")

    # Training settings
    parser.add_argument("--discount", type=float, default=None, help="Discount factor gamma")
    parser.add_argument("--lambda_", type=float, default=None, help="Lambda for TD(lambda)")
    parser.add_argument("--entropy_scale", type=float, default=None, help="Entropy bonus scale")
    parser.add_argument("--no_advantage_normalization", action="store_true",
                        help="Disable advantage normalization (use raw advantages)")
    parser.add_argument("--free_nats", type=float, default=None, help="KL free nats threshold")
    parser.add_argument("--beta_prior", type=float, default=None, help="Prior KL weight")
    parser.add_argument("--beta_posterior", type=float, default=None, help="Posterior KL weight")
    parser.add_argument("--gradient_clip", type=float, default=None, help="Gradient clipping")
    parser.add_argument("--uniform_mix", type=float, default=None,
                        help="Uniform mixing ratio for latent categoricals (prevents collapse, default: 0.01)")

    # Buffer
    parser.add_argument("--buffer_capacity", type=int, default=None, help="Replay buffer capacity")

    # Checkpointing and evaluation
    parser.add_argument("--checkpoint_interval", type=int, default=None,
                        help="Checkpoint save interval (gradient steps)")
    parser.add_argument("--eval_interval", type=int, default=None,
                        help="Evaluation interval (gradient steps)")
    parser.add_argument("--eval_episodes", type=int, default=None, help="Episodes per evaluation")

    # GIF recording
    parser.add_argument("--gif_interval", type=int, default=None,
                        help="GIF recording interval (gradient steps, 0=disabled)")
    parser.add_argument("--gif_episodes", type=int, default=3, help="Episodes per GIF")

    # Logging
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--log_interval", type=int, default=10, help="Console log interval")

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")

    # Self-play settings
    parser.add_argument("--self_play_start", type=int, default=None,
                        help="Episode to activate self-play (default: disabled)")
    parser.add_argument("--self_play_pool_size", type=int, default=10,
                        help="Number of past checkpoints to keep in self-play pool")
    parser.add_argument("--self_play_save_interval", type=int, default=500,
                        help="Episodes between adding new opponents to pool")
    parser.add_argument("--self_play_weak_ratio", type=float, default=0.3,
                        help="Probability of training against anchor (weak/strong) vs pool")
    parser.add_argument("--use_pfsp", action="store_true",
                        help="Enable Prioritized Fictitious Self-Play opponent selection")
    parser.add_argument("--pfsp_mode", type=str, default="variance",
                        choices=["variance", "hard", "uniform"],
                        help="PFSP mode: variance (50%% winrate), hard (lowest winrate)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = loadConfig(args.config)

    # Apply CLI overrides to config
    if args.seed is not None:
        config.seed = args.seed
    if args.opponent is not None:
        config.opponent = args.opponent
    if args.run_name is not None:
        config.runName = args.run_name

    # Training schedule overrides
    if args.gradient_steps is not None:
        config.gradientSteps = args.gradient_steps
    if args.replay_ratio is not None:
        config.replayRatio = args.replay_ratio
    if args.warmup_episodes is not None:
        config.episodesBeforeStart = args.warmup_episodes
    if args.interaction_episodes is not None:
        config.numInteractionEpisodes = args.interaction_episodes

    # DreamerV3 hyperparameter overrides
    if args.batch_size is not None:
        config.dreamer.batchSize = args.batch_size
    if args.batch_length is not None:
        config.dreamer.batchLength = args.batch_length
    if args.imagination_horizon is not None:
        config.dreamer.imaginationHorizon = args.imagination_horizon
    if args.recurrent_size is not None:
        config.dreamer.recurrentSize = args.recurrent_size
    if args.latent_length is not None:
        config.dreamer.latentLength = args.latent_length
    if args.latent_classes is not None:
        config.dreamer.latentClasses = args.latent_classes
    if args.encoded_obs_size is not None:
        config.dreamer.encodedObsSize = args.encoded_obs_size

    # Learning rate overrides
    if args.lr_world is not None:
        config.dreamer.worldModelLR = args.lr_world
    if args.lr_actor is not None:
        config.dreamer.actorLR = args.lr_actor
    if args.lr_critic is not None:
        config.dreamer.criticLR = args.lr_critic

    # Training settings overrides
    if args.discount is not None:
        config.dreamer.discount = args.discount
    if args.lambda_ is not None:
        config.dreamer.lambda_ = args.lambda_
    if args.entropy_scale is not None:
        config.dreamer.entropyScale = args.entropy_scale
    if args.no_advantage_normalization:
        config.dreamer.useAdvantageNormalization = False
    if args.free_nats is not None:
        config.dreamer.freeNats = args.free_nats
    if args.beta_prior is not None:
        config.dreamer.betaPrior = args.beta_prior
    if args.beta_posterior is not None:
        config.dreamer.betaPosterior = args.beta_posterior
    if args.gradient_clip is not None:
        config.dreamer.gradientClip = args.gradient_clip
    if args.uniform_mix is not None:
        # Apply to both prior and posterior networks
        config.dreamer.priorNet.uniformMix = args.uniform_mix
        config.dreamer.posteriorNet.uniformMix = args.uniform_mix

    # Buffer override
    if args.buffer_capacity is not None:
        config.dreamer.buffer.capacity = args.buffer_capacity

    # Checkpoint/eval overrides
    if args.checkpoint_interval is not None:
        config.checkpointInterval = args.checkpoint_interval
    if args.eval_interval is not None:
        config.evalInterval = args.eval_interval
    if args.eval_episodes is not None:
        config.numEvaluationEpisodes = args.eval_episodes

    # GIF overrides
    if args.gif_interval is not None:
        config.gifInterval = args.gif_interval
    else:
        config.gifInterval = config.get('gifInterval', 10000)  # Default 10k
    config.gifEpisodes = args.gif_episodes

    # W&B overrides
    if args.wandb_project is not None:
        config.wandbProject = args.wandb_project
    if args.wandb_entity is not None:
        config.wandbEntity = args.wandb_entity

    # Set seed
    seedEverything(config.seed)

    # Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")

    # Create run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runName = f"{config.runName}_{config.opponent}_seed{config.seed}_{timestamp}"

    # Setup paths
    checkpointFilenameBase = os.path.join(config.folderNames.checkpointsFolder, runName)
    metricsFilename = os.path.join(config.folderNames.metricsFolder, runName)
    plotFilename = os.path.join(config.folderNames.plotsFolder, runName)
    ensureParentFolders(checkpointFilenameBase, metricsFilename, plotFilename)

    # Initialize W&B
    use_wandb = config.get('useWandB', True) and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=config.get('wandbProject', 'rl-hockey'),
            entity=config.get('wandbEntity', None),
            name=runName,
            config={
                "algorithm": "DreamerV3",
                "seed": config.seed,
                "opponent": config.opponent,
                "batch_size": config.dreamer.batchSize,
                "batch_length": config.dreamer.batchLength,
                "imagination_horizon": config.dreamer.imaginationHorizon,
                "recurrent_size": config.dreamer.recurrentSize,
                "latent_length": config.dreamer.latentLength,
                "latent_classes": config.dreamer.latentClasses,
                "actor_lr": config.dreamer.actorLR,
                "critic_lr": config.dreamer.criticLR,
                "world_model_lr": config.dreamer.worldModelLR,
                "discount": config.dreamer.discount,
                "lambda": config.dreamer.lambda_,
                "entropy_scale": config.dreamer.entropyScale,
                "free_nats": config.dreamer.freeNats,
                "buffer_capacity": config.dreamer.buffer.capacity,
                "uniform_mix": config.dreamer.priorNet.uniformMix,
                "use_auxiliary_tasks": config.get('useAuxiliaryTasks', True),
                "gif_interval": config.gifInterval,
                # Self-play settings
                "self_play_start": args.self_play_start,
                "self_play_pool_size": args.self_play_pool_size,
                "self_play_save_interval": args.self_play_save_interval,
                "self_play_weak_ratio": args.self_play_weak_ratio,
                "use_pfsp": args.use_pfsp,
                "pfsp_mode": args.pfsp_mode,
            }
        )

    # Create environment
    mode_map = {
        'NORMAL': Mode.NORMAL,
        'TRAIN_SHOOTING': Mode.TRAIN_SHOOTING,
        'TRAIN_DEFENSE': Mode.TRAIN_DEFENSE,
    }
    env_mode = mode_map.get(args.mode, Mode.NORMAL)
    env = h_env.HockeyEnv(mode=env_mode, keep_mode=True)
    observationSize = env.observation_space.shape[0]
    actionSize = env.action_space.shape[0] // 2  # Only agent's actions (4)
    actionLow = env.action_space.low[:actionSize].tolist()
    actionHigh = env.action_space.high[:actionSize].tolist()

    print(f"Hockey environment: obs={observationSize}, action={actionSize}, mode={args.mode}")
    print(f"Action bounds: low={actionLow}, high={actionHigh}")

    # Create training opponent (initial)
    opponent = create_opponent(config.opponent)
    print(f"Initial opponent: {config.opponent}")

    # Create evaluation opponents (always have both for proper evaluation)
    eval_opponent_weak = create_opponent("weak")
    eval_opponent_strong = create_opponent("strong")

    # Create agent
    agent = Dreamer(observationSize, actionSize, actionLow, actionHigh, device, config.dreamer)

    # Setup self-play manager (if enabled)
    self_play_enabled = args.self_play_start is not None
    self_play_manager = None

    if self_play_enabled:
        self_play_manager = SelfPlayManager(
            pool_size=args.self_play_pool_size,
            save_interval=args.self_play_save_interval,
            weak_ratio=args.self_play_weak_ratio,
            device=device,
            use_pfsp=args.use_pfsp,
            pfsp_mode=args.pfsp_mode,
            agent_class=Dreamer,
            obs_size=observationSize,
            action_size=actionSize,
            action_low=actionLow,
            action_high=actionHigh,
            config=config.dreamer,
        )
        print(f"\nSelf-play enabled:")
        print(f"  Activation episode: {args.self_play_start}")
        print(f"  Pool size: {args.self_play_pool_size}")
        print(f"  Save interval: {args.self_play_save_interval}")
        print(f"  Weak ratio: {args.self_play_weak_ratio}")
        print(f"  PFSP: {args.use_pfsp} (mode: {args.pfsp_mode})")

    # Resume if requested
    if args.resume:
        if os.path.exists(args.resume) or os.path.exists(args.resume + '.pth'):
            agent.loadCheckpoint(args.resume)
            print(f"Resumed from: {args.resume}")
        else:
            print(f"Warning: Checkpoint not found: {args.resume}")

    # Print config summary
    print(f"\n=== Configuration ===")
    print(f"Gradient steps: {config.gradientSteps}")
    print(f"Replay ratio: {config.replayRatio}")
    print(f"Batch: {config.dreamer.batchSize} x {config.dreamer.batchLength}")
    print(f"Imagination horizon: {config.dreamer.imaginationHorizon}")
    print(f"LR: world={config.dreamer.worldModelLR}, actor={config.dreamer.actorLR}, critic={config.dreamer.criticLR}")
    print(f"Entropy scale: {config.dreamer.entropyScale}")
    print(f"GIF interval: {config.gifInterval}")
    print()

    # === Warmup Phase ===
    print(f"=== Warmup: {config.episodesBeforeStart} episodes ===")
    for ep in range(config.episodesBeforeStart):
        _, steps, outcome, transitions, _ = run_episode(
            env, agent, opponent, training=True, seed=config.seed + ep
        )
        for trans in transitions:
            agent.buffer.add(*trans)
        print(f"Warmup {ep+1}/{config.episodesBeforeStart}: {steps} steps, {outcome}")
        agent.totalEpisodes += 1

    print(f"Buffer size after warmup: {len(agent.buffer)}")

    # === Main Training Loop ===
    print(f"\n=== Training: {config.gradientSteps} gradient steps ===")

    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    episode_outcomes = {'win': 0, 'loss': 0, 'draw': 0}
    recent_wins = []
    recent_lengths = []
    start_time = time.time()

    iterationsNum = config.gradientSteps // config.replayRatio
    for iteration in range(iterationsNum):
        # === Gradient Updates ===
        worldModelMetrics = {}
        behaviorMetrics = {}

        for _ in range(config.replayRatio):
            sampledData = agent.buffer.sample(agent.config.batchSize, agent.config.batchLength)
            initialStates, wmMetrics = agent.worldModelTraining(sampledData)
            bhMetrics = agent.behaviorTraining(initialStates)
            agent.totalGradientSteps += 1

            worldModelMetrics = wmMetrics
            behaviorMetrics = bhMetrics

        # === Environment Interaction ===
        for _ in range(config.numInteractionEpisodes):
            # === Self-Play Opponent Selection ===
            current_opponent = opponent  # Default to initial opponent
            current_opponent_type = config.opponent

            if self_play_enabled and self_play_manager is not None:
                # Check for activation
                if self_play_manager.should_activate(agent.totalEpisodes, args.self_play_start):
                    selfplay_dir = os.path.join(config.folderNames.checkpointsFolder, runName)
                    self_play_manager.activate(agent.totalEpisodes, selfplay_dir, agent)

                # Select opponent
                if self_play_manager.active:
                    current_opponent_type = self_play_manager.select_opponent()

                    if current_opponent_type == 'weak':
                        current_opponent = eval_opponent_weak
                    elif current_opponent_type == 'strong':
                        current_opponent = eval_opponent_strong
                    elif current_opponent_type == 'self-play':
                        sp_opponent = self_play_manager.get_opponent()
                        if sp_opponent is not None:
                            current_opponent = sp_opponent
                        else:
                            current_opponent = eval_opponent_weak
                            current_opponent_type = 'weak'

                    # Reset opponent state for new episode
                    self_play_manager.reset_opponent()

            # Run episode
            current_opponent.reset()
            reward, steps, outcome, transitions, _ = run_episode(
                env, agent, current_opponent, training=True
            )

            for trans in transitions:
                agent.buffer.add(*trans)

            agent.totalEnvSteps += steps
            agent.totalEpisodes += 1

            # Record result for PFSP
            if self_play_enabled and self_play_manager is not None and self_play_manager.active:
                winner = 1 if outcome == 'win' else (-1 if outcome == 'loss' else 0)
                self_play_manager.record_result(winner)

                # Update pool periodically
                self_play_manager.update_pool(
                    agent.totalEpisodes, agent,
                    os.path.join(config.folderNames.checkpointsFolder, runName)
                )

            # Track episode metrics
            episode_rewards.append(reward)
            episode_lengths.append(steps)
            episode_outcomes[outcome] += 1

            # Rolling windows for recent stats
            recent_wins.append(1 if outcome == 'win' else 0)
            recent_lengths.append(steps)
            if len(recent_wins) > 100:
                recent_wins.pop(0)
                recent_lengths.pop(0)

        # === Console Logging ===
        if iteration % args.log_interval == 0:
            elapsed = time.time() - start_time
            win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
            mean_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            mean_length = np.mean(recent_lengths) if recent_lengths else 0

            # Get entropy from behavior metrics (handle renamed key)
            entropy_val = behaviorMetrics.get('behavior/entropy_mean', behaviorMetrics.get('behavior/entropy', 0))

            print(f"[{agent.totalGradientSteps:>7}] "
                  f"Ep={agent.totalEpisodes:>5} | "
                  f"Steps={agent.totalEnvSteps:>7} | "
                  f"Win={win_rate:.1%} | "
                  f"R={mean_reward:>6.2f} | "
                  f"WM={worldModelMetrics.get('world/loss', 0):.3f} | "
                  f"Act={behaviorMetrics.get('behavior/actor_loss', 0):.3f} | "
                  f"Crit={behaviorMetrics.get('behavior/critic_loss', 0):.3f} | "
                  f"Ent={entropy_val:.2f}")

            if use_wandb:
                log_dict = {
                    # Core stats
                    "stats/gradient_steps": agent.totalGradientSteps,
                    "stats/env_steps": agent.totalEnvSteps,
                    "stats/episodes": agent.totalEpisodes,
                    "stats/win_rate": win_rate,
                    "stats/mean_reward": mean_reward,
                    "stats/wins": episode_outcomes['win'],
                    "stats/losses": episode_outcomes['loss'],
                    "stats/draws": episode_outcomes['draw'],
                    "stats/buffer_size": len(agent.buffer),

                    # Episode-level stats
                    "episode/length_mean": mean_length,
                    "episode/length_std": np.std(recent_lengths) if len(recent_lengths) > 1 else 0,
                    "episode/length_min": min(recent_lengths) if recent_lengths else 0,
                    "episode/length_max": max(recent_lengths) if recent_lengths else 0,
                    "episode/reward_mean": mean_reward,
                    "episode/reward_std": np.std(episode_rewards[-100:]) if len(episode_rewards) > 1 else 0,

                    # Outcome rates (last 100 episodes)
                    "episode/win_rate_100": win_rate,
                    "episode/draw_rate_100": recent_lengths.count(250) / len(recent_lengths) if recent_lengths else 0,  # Max length = draw

                    # Timing
                    "time/elapsed_hours": elapsed / 3600,
                    "time/steps_per_second": agent.totalEnvSteps / elapsed if elapsed > 0 else 0,
                    "time/episodes_per_hour": agent.totalEpisodes / (elapsed / 3600) if elapsed > 0 else 0,
                    "time/gradient_steps_per_second": agent.totalGradientSteps / elapsed if elapsed > 0 else 0,
                }
                log_dict.update(worldModelMetrics)
                log_dict.update(behaviorMetrics)

                # Add self-play metrics
                if self_play_enabled and self_play_manager is not None:
                    log_dict.update(self_play_manager.get_stats())

                wandb.log(log_dict, step=agent.totalGradientSteps)

        # === Checkpointing ===
        if agent.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
            suffix = f"{agent.totalGradientSteps // 1000}k"
            checkpoint_path = f"{checkpointFilenameBase}_{suffix}"
            agent.saveCheckpoint(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            if use_wandb:
                wandb.save(checkpoint_path + '.pth')

        # === Evaluation (against both weak and strong) ===
        if agent.totalGradientSteps % config.get('evalInterval', 1000) == 0:
            eval_metrics = {}

            # Evaluate against weak opponent
            weak_metrics = evaluate_against_opponent(
                env, agent, eval_opponent_weak,
                config.numEvaluationEpisodes, "weak"
            )
            eval_metrics.update(weak_metrics)

            # Evaluate against strong opponent
            strong_metrics = evaluate_against_opponent(
                env, agent, eval_opponent_strong,
                config.numEvaluationEpisodes, "strong"
            )
            eval_metrics.update(strong_metrics)

            # Compute combined metrics
            weak_wr = weak_metrics['eval/weak_win_rate']
            strong_wr = strong_metrics['eval/strong_win_rate']
            eval_metrics['eval/combined_win_rate'] = (weak_wr + strong_wr) / 2

            print(f"  EVAL: Weak={weak_wr:.1%}, Strong={strong_wr:.1%}, Combined={(weak_wr + strong_wr)/2:.1%}")

            if use_wandb:
                wandb.log(eval_metrics, step=agent.totalGradientSteps)

        # === GIF Recording (against both weak and strong) ===
        if config.gifInterval > 0 and agent.totalGradientSteps % config.gifInterval == 0 and use_wandb:
            # Record GIF against weak opponent
            record_and_log_gif(
                env, agent, eval_opponent_weak, "weak",
                agent.totalGradientSteps, config.gifEpisodes, use_wandb
            )

            # Record GIF against strong opponent
            record_and_log_gif(
                env, agent, eval_opponent_strong, "strong",
                agent.totalGradientSteps, config.gifEpisodes, use_wandb
            )

        # === Save Metrics ===
        if config.saveMetrics and iteration % 100 == 0:
            metricsBase = {
                "envSteps": agent.totalEnvSteps,
                "gradientSteps": agent.totalGradientSteps,
                "episodes": agent.totalEpisodes,
                "winRate": sum(recent_wins) / len(recent_wins) if recent_wins else 0,
                "meanReward": np.mean(episode_rewards[-100:]) if episode_rewards else 0,
            }
            saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics)

    # === Final Save ===
    final_path = f"{checkpointFilenameBase}_final"
    agent.saveCheckpoint(final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")

    if config.saveMetrics:
        plotMetrics(metricsFilename, savePath=plotFilename, title=f"DreamerV3 Hockey - {runName}")

    if use_wandb:
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
