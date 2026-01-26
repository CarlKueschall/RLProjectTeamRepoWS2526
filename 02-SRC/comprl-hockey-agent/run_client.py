"""
COMPRL Tournament Client for DreamerV3 Hockey Agent.

This client connects trained DreamerV3 agents to the competition server.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

from __future__ import annotations

import argparse
import uuid
import sys
import os
import torch
import yaml

# Force unbuffered output for immediate logging
if sys.stdout.isatty():
    pass
else:
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

print("=" * 70, flush=True)
print("TOURNAMENT CLIENT: Script starting...", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)
print("=" * 70, flush=True)

import hockey.hockey_env as h_env
import numpy as np

# Add parent directory (02-SRC) to path so we can import DreamerV3
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
dreamer_dir = os.path.join(parent_dir, 'DreamerV3')
sys.path.insert(0, parent_dir)
sys.path.insert(0, dreamer_dir)

from comprl.client import Agent, launch_client


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()
        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


# Default checkpoint path for DreamerV3 agent
# Can be overridden via DREAMER_CHECKPOINT environment variable
DEFAULT_DREAMER_CHECKPOINT = os.environ.get(
    'DREAMER_CHECKPOINT',
    os.path.join(
        script_dir,
        "checkpoint_266k.pth"
    )
)


class DreamerV3Config:
    """Minimal config object for DreamerV3 inference."""

    def __init__(self):
        # State dimensions
        self.recurrentSize = 256
        self.latentLength = 16
        self.latentClasses = 16
        self.encodedObsSize = 256

        # Network architectures
        self.encoder = type('Config', (), {
            'hiddenSize': 256,
            'numLayers': 2,
            'activation': 'Tanh'
        })()

        self.decoder = type('Config', (), {
            'hiddenSize': 256,
            'numLayers': 2,
            'activation': 'Tanh'
        })()

        self.recurrentModel = type('Config', (), {
            'hiddenSize': 256,
            'activation': 'Tanh'
        })()

        self.priorNet = type('Config', (), {
            'hiddenSize': 256,
            'numLayers': 2,
            'activation': 'Tanh',
            'uniformMix': 0.01
        })()

        self.posteriorNet = type('Config', (), {
            'hiddenSize': 256,
            'numLayers': 2,
            'activation': 'Tanh',
            'uniformMix': 0.01
        })()

        self.reward = type('Config', (), {
            'hiddenSize': 256,
            'numLayers': 2,
            'activation': 'Tanh'
        })()

        self.continuation = type('Config', (), {
            'hiddenSize': 256,
            'numLayers': 2,
            'activation': 'Tanh'
        })()

        self.actor = type('Config', (), {
            'hiddenSize': 256,
            'numLayers': 2,
            'activation': 'Tanh'
        })()

        self.critic = type('Config', (), {
            'hiddenSize': 256,
            'numLayers': 2,
            'activation': 'Tanh'
        })()

        # Buffer config (minimal, not used for inference)
        self.buffer = type('Config', (), {
            'capacity': 1000,
            'useDreamSmooth': False,
            'dreamsmoothAlpha': 0.5
        })()

        # Other settings
        self.useContinuationPrediction = True
        self.useAuxiliaryTasks = True
        self.auxHiddenSize = 128
        self.goalPredictionHorizon = 25
        self.goalRewardThreshold = 1.0
        self.auxTaskScale = 1.0
        self.slowCriticDecay = 0.98

        # Learning rates (not used for inference but needed for optimizer init)
        self.worldModelLR = 0.0003
        self.actorLR = 0.0001
        self.criticLR = 0.0001

        # Two-hot encoding
        self.twoHotBins = 255
        self.twoHotMinVal = -20.0
        self.twoHotMaxVal = 20.0


class DreamerV3HockeyAgent(Agent):
    """A hockey agent using the DreamerV3 reinforcement learning algorithm."""

    def __init__(self, model_path: str) -> None:
        super().__init__()

        print("=" * 70)
        print("DreamerV3HockeyAgent: Initializing...")
        print("=" * 70)

        # Set device
        self.device = torch.device('cpu')  # Use CPU for tournament (consistent behavior)
        print(f"[INIT] Using device: {self.device}")

        # Create minimal config for inference
        print("[INIT] Creating DreamerV3 config...")
        config = DreamerV3Config()

        # Import Dreamer (after adding path)
        print("[INIT] Importing Dreamer agent...")
        from dreamer import Dreamer

        # Initialize hockey environment to get action bounds
        print("[INIT] Creating hockey environment for action bounds...")
        env = h_env.HockeyEnv()

        # Agent uses 4-dim actions (own player only)
        observation_size = 18
        action_size = 4
        action_low = env.action_space.low[:4]
        action_high = env.action_space.high[:4]

        print(f"[INIT] Observation size: {observation_size}")
        print(f"[INIT] Action size: {action_size}")
        print(f"[INIT] Action bounds: [{action_low[0]:.2f}, {action_high[0]:.2f}]")

        # Create Dreamer agent
        print("[INIT] Creating Dreamer agent...")
        self.agent = Dreamer(
            observationSize=observation_size,
            actionSize=action_size,
            actionLow=action_low,
            actionHigh=action_high,
            device=self.device,
            config=config
        )

        # Load checkpoint
        if model_path:
            # Handle both absolute and relative paths
            if not os.path.isabs(model_path):
                model_path = os.path.join(script_dir, model_path)
            model_path = os.path.abspath(model_path)

            print(f"[INIT] Loading checkpoint: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Load network weights
            self._load_checkpoint(checkpoint)
            print("[INIT] ✓ Checkpoint loaded successfully!")

        env.close()

        # Initialize recurrent state
        self.h = None
        self.z = None
        self._step_count = 0

        print("=" * 70)
        print("DreamerV3HockeyAgent: Initialization complete. Ready to play!")
        print("=" * 70)

    def _load_checkpoint(self, checkpoint: dict) -> None:
        """Load network weights from checkpoint."""

        # Load world model components
        self.agent.encoder.load_state_dict(checkpoint['encoder'])
        self.agent.decoder.load_state_dict(checkpoint['decoder'])
        self.agent.recurrentModel.load_state_dict(checkpoint['recurrentModel'])
        self.agent.priorNet.load_state_dict(checkpoint['priorNet'])
        self.agent.posteriorNet.load_state_dict(checkpoint['posteriorNet'])
        self.agent.rewardPredictor.load_state_dict(checkpoint['rewardPredictor'])

        # Load actor and critic
        self.agent.actor.load_state_dict(checkpoint['actor'])
        self.agent.critic.load_state_dict(checkpoint['critic'])

        # Load slow critic if available
        if 'slowCritic' in checkpoint:
            self.agent.slowCritic.load_state_dict(checkpoint['slowCritic'])

        # Load continue predictor if available
        if 'continuePredictor' in checkpoint:
            self.agent.continuePredictor.load_state_dict(checkpoint['continuePredictor'])

        # Load auxiliary task heads if available
        if 'goalPredictor' in checkpoint:
            self.agent.goalPredictor.load_state_dict(checkpoint['goalPredictor'])
        if 'puckGoalDistPredictor' in checkpoint:
            self.agent.puckGoalDistPredictor.load_state_dict(checkpoint['puckGoalDistPredictor'])
        if 'shotQualityPredictor' in checkpoint:
            self.agent.shotQualityPredictor.load_state_dict(checkpoint['shotQualityPredictor'])

        # Get training stats if available
        if 'totalEpisodes' in checkpoint:
            print(f"[LOAD] Checkpoint trained for {checkpoint['totalEpisodes']} episodes, "
                  f"{checkpoint.get('totalGradientSteps', 'N/A')} gradient steps")

    def get_step(self, observation: list[float]) -> list[float]:
        """Get action for the given observation."""

        self._step_count += 1

        # Convert list to numpy array
        obs_array = np.array(observation, dtype=np.float32)

        if self._step_count <= 5 or self._step_count % 100 == 0:
            print(f"[STEP {self._step_count}] Observation shape: {len(obs_array)}")

        # Diagnostic on first step
        if not hasattr(self, '_obs_diagnostic_done'):
            self._print_observation_diagnostic(obs_array)
            self._obs_diagnostic_done = True

        # Handle observation dimension mismatch
        original_dim = len(obs_array)
        if len(obs_array) == 16:
            # Pad with zeros to match 18-dim expected input (keep_mode=OFF case)
            obs_array = np.pad(obs_array, (0, 2), mode='constant', constant_values=0.0)
            if self._step_count <= 5:
                print(f"[STEP {self._step_count}] Padded 16-dim to 18-dim")
        elif len(obs_array) > 18:
            # Truncate if longer than expected
            obs_array = obs_array[:18]
            if self._step_count <= 5:
                print(f"[STEP {self._step_count}] Truncated from {original_dim} to 18-dim")

        # Get action from DreamerV3 agent (maintains recurrent state)
        action, self.h, self.z = self.agent.act(obs_array, self.h, self.z)

        # Ensure action is in valid range [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        # Validate action
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f"[STEP {self._step_count}] ⚠ WARNING: Invalid action detected, using zeros")
            action = np.zeros(4)

        if self._step_count <= 5:
            print(f"[STEP {self._step_count}] Action: {action}")

        return action.tolist()

    def _print_observation_diagnostic(self, obs_array: np.ndarray) -> None:
        """Print diagnostic information about the observation."""
        print(f"\n{'='*70}")
        print(f"DREAMERV3 AGENT DIAGNOSTIC: First Observation Analysis")
        print(f"{'='*70}")
        print(f"Observation dimensions: {len(obs_array)}")

        if len(obs_array) == 16:
            print(f"  → Tournament uses keep_mode=OFF (16-dim observations)")
        elif len(obs_array) == 18:
            print(f"  → Tournament uses keep_mode=ON (18-dim observations)")
        else:
            print(f"  → Unexpected observation dimension: {len(obs_array)}")

        print(f"\nObservation breakdown (assuming hockey env):")
        print(f"  Player 1 pos (0-1):    {obs_array[0:2]}")
        print(f"  Player 1 angle (2):    {obs_array[2]:.4f}")
        print(f"  Player 1 vel (3-4):    {obs_array[3:5]}")
        print(f"  Player 1 ang vel (5):  {obs_array[5]:.4f}")
        print(f"  Player 2 pos (6-7):    {obs_array[6:8]}")
        print(f"  Player 2 angle (8):    {obs_array[8]:.4f}")
        print(f"  Player 2 vel (9-10):   {obs_array[9:11]}")
        print(f"  Player 2 ang vel (11): {obs_array[11]:.4f}")
        print(f"  Puck pos (12-13):      {obs_array[12:14]}")
        print(f"  Puck vel (14-15):      {obs_array[14:16]}")
        if len(obs_array) > 16:
            print(f"  Extra (16-17):         {obs_array[16:18]}")

        # Perspective detection
        player1_x = obs_array[0]
        if player1_x < 0:
            print(f"\n  ✓ NORMAL perspective (Player 1 on left)")
        elif player1_x > 0:
            print(f"\n  ⚠ MIRRORED perspective (Player 1 on right)")
        else:
            print(f"\n  ? Cannot determine perspective")

        print(f"{'='*70}\n")

    def on_start_game(self, game_id) -> None:
        """Called when a new game starts."""
        game_id = uuid.UUID(int=int.from_bytes(game_id, byteorder='big'))
        print("=" * 70)
        print(f"🎮 GAME STARTED (id: {game_id})")
        print("=" * 70)

        # IMPORTANT: Reset recurrent state for new game
        self.h = None
        self.z = None
        self._step_count = 0

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        """Called when a game ends."""
        text_result = "WON ✓" if result else "LOST ✗"
        print(
            f"Game ended: {text_result} | My score: {stats[0]:.1f} | "
            f"Opponent score: {stats[1]:.1f} | Steps: {self._step_count}"
        )


def initialize_agent(agent_args: list[str]) -> Agent:
    """Initialize the agent based on command-line arguments."""

    print("\n" + "=" * 70)
    print("TOURNAMENT CLIENT: Initializing agent...")
    print("=" * 70)
    print(f"Arguments: {agent_args}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "dreamer"],
        default="dreamer",
        help="Which agent to use.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_DREAMER_CHECKPOINT,
        help="Path to trained DreamerV3 model checkpoint",
    )
    args = parser.parse_args(agent_args)

    print(f"Agent type: {args.agent}")
    if args.model_path:
        print(f"Model path: {args.model_path}")

    # Initialize the agent based on the arguments
    agent: Agent
    if args.agent == "weak":
        print("Creating weak HockeyAgent...")
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        print("Creating strong HockeyAgent...")
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        print("Creating RandomAgent...")
        agent = RandomAgent()
    elif args.agent == "dreamer":
        print("Creating DreamerV3HockeyAgent...")
        agent = DreamerV3HockeyAgent(model_path=args.model_path)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    print("=" * 70)
    print("Agent initialized successfully!")
    print("=" * 70 + "\n")

    return agent


def main() -> None:
    print("\n" + "=" * 70, flush=True)
    print("TOURNAMENT CLIENT: Starting connection...", flush=True)
    print("=" * 70, flush=True)
    print(f"Server URL: {os.environ.get('COMPRL_SERVER_URL', 'NOT SET')}", flush=True)
    print(f"Server Port: {os.environ.get('COMPRL_SERVER_PORT', 'NOT SET')}", flush=True)
    print(f"Access Token: {'SET' if os.environ.get('COMPRL_ACCESS_TOKEN') else 'NOT SET'}", flush=True)
    print("=" * 70 + "\n", flush=True)

    print("Connecting to tournament server...", flush=True)
    print("(Waiting for games to be assigned...)\n", flush=True)

    try:
        launch_client(initialize_agent)
    except Exception as e:
        print(f"\n{'='*70}", flush=True)
        print(f"ERROR: {e}", flush=True)
        print(f"{'='*70}\n", flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("TOURNAMENT CLIENT: Entering main()", flush=True)
    print("=" * 70, flush=True)
    main()
