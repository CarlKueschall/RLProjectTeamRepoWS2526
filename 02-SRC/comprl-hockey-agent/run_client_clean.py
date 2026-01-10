"""
Clean tournament client for TD3 hockey agent.
Matches test_hockey.py behavior as closely as possible for maximum performance.
"""
from __future__ import annotations

import argparse
import uuid
import sys
import os
import torch
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import hockey.hockey_env as h_env

# Add parent directory to path for TD3 imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
from TD3.agents.td3_agent import TD3Agent

from comprl.client import Agent, launch_client


class TD3HockeyAgent(Agent):
    """
    Tournament-ready TD3 agent.
    Designed to match test_hockey.py behavior exactly.
    """

    def __init__(self, model_path: str, hidden_actor: list = None, hidden_critic: list = None) -> None:
        super().__init__()

        print("=" * 60)
        print("TD3HockeyAgent: Initializing...")

        # Create environment to get spaces (with keep_mode=True to match training)
        env = h_env.HockeyEnv(keep_mode=True)

        # Load checkpoint to detect architecture
        print(f"Loading checkpoint: {model_path}")

        if not os.path.isabs(model_path):
            model_path = os.path.join(script_dir, model_path)
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Extract agent state
        if isinstance(checkpoint, dict) and 'agent_state' in checkpoint:
            agent_state = checkpoint['agent_state']
        else:
            agent_state = checkpoint

        # Detect critic action dimension from checkpoint
        critic_action_dim = 8  # Default: new version
        if isinstance(agent_state, tuple) and len(agent_state) >= 1:
            q1_state = agent_state[0]
            if 'layers.0.weight' in q1_state:
                critic_input_dim = q1_state['layers.0.weight'].shape[1]
                if critic_input_dim == 22:
                    critic_action_dim = 4  # Old version: 18 obs + 4 actions
                elif critic_input_dim == 26:
                    critic_action_dim = 8  # New version: 18 obs + 8 actions
                print(f"Detected critic input dim: {critic_input_dim} (action_dim={critic_action_dim})")

        # Use provided hidden sizes or defaults
        if hidden_actor is None:
            hidden_actor = [256, 256]
        if hidden_critic is None:
            hidden_critic = [256, 256, 128]

        print(f"Network architecture: actor={hidden_actor}, critic={hidden_critic}")

        # Create action space matching test_hockey.py exactly
        from gymnasium import spaces
        agent_action_space = spaces.Box(
            low=env.action_space.low[:4],
            high=env.action_space.high[:4],
            dtype=env.action_space.dtype
        )

        # Store action bounds for reference
        self.action_low = env.action_space.low[:4]
        self.action_high = env.action_space.high[:4]
        print(f"Action bounds: [{self.action_low[0]:.2f}, {self.action_high[0]:.2f}]")

        # Create TD3 agent (matching test_hockey.py)
        self.agent = TD3Agent(
            observation_space=env.observation_space,  # 18-dim with keep_mode=True
            action_space=agent_action_space,          # 4-dim
            hidden_sizes_actor=hidden_actor,
            hidden_sizes_critic=hidden_critic,
            critic_action_dim=critic_action_dim,
        )

        # Fix state dict keys (readout -> output_layer)
        if isinstance(agent_state, tuple):
            fixed_state = []
            for state_dict in agent_state:
                fixed_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('readout.', 'output_layer.')
                    fixed_dict[new_key] = value
                fixed_state.append(fixed_dict)
            agent_state = tuple(fixed_state)

        # Restore agent state
        self.agent.restore_state(agent_state)
        print("Checkpoint loaded successfully!")

        # CRITICAL: Put policy network in evaluation mode!
        # This ensures BatchNorm uses running statistics, not batch statistics
        self.agent.policy.eval()
        self.agent.policy_target.eval()
        self.agent.Q1.eval()
        self.agent.Q2.eval()
        self.agent.Q1_target.eval()
        self.agent.Q2_target.eval()
        print("Networks set to EVAL mode (BatchNorm uses running stats)")

        env.close()
        print("=" * 60)
        print("Ready for tournament!")
        print("=" * 60 + "\n")

        self._game_count = 0
        self._step_count = 0

    def get_step(self, observation: list[float]) -> list[float]:
        """
        Get action for given observation.
        Matches test_hockey.py behavior exactly.
        """
        # Convert to numpy array
        obs = np.array(observation, dtype=np.float32)

        # Handle observation dimension
        # Training used 18-dim (keep_mode=True)
        # Tournament might send 16-dim or 18-dim
        if len(obs) == 16:
            # Pad with zeros for keep_time values
            obs = np.pad(obs, (0, 2), mode='constant', constant_values=0.0)
        elif len(obs) > 18:
            obs = obs[:18]

        # Get action from agent with NO exploration noise (eps=0.0)
        # This matches test_hockey.py: test_eps = 0.0 if args.deterministic else 0.1
        action = self.agent.act(obs, eps=0.0)

        # Take first 4 dimensions (agent's own actions)
        # This matches test_hockey.py: action1[:4]
        action = action[:4]

        # NO extra clipping! TD3Agent.act() already clips to action_space bounds
        # test_hockey.py doesn't do extra clipping either

        # Validate action (safety check)
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f"WARNING: Invalid action detected, using zeros")
            action = np.zeros(4)

        self._step_count += 1
        return action.tolist()

    def on_start_game(self, game_id) -> None:
        """Called when a new game starts."""
        self._game_count += 1
        self._step_count = 0

        # Reset agent (matches test_hockey.py: agent.reset())
        # This resets the OUNoise state
        self.agent.reset()

        game_uuid = uuid.UUID(int=int.from_bytes(game_id, byteorder='big'))
        print(f"Game {self._game_count} started (id: {game_uuid})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        """Called when game ends."""
        outcome = "WON" if result else "LOST"
        print(f"Game {self._game_count} {outcome}: {stats[0]:.0f} - {stats[1]:.0f} ({self._step_count} steps)")


def initialize_agent(agent_args: list[str]) -> Agent:
    """Initialize agent from command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["weak", "strong", "random", "td3"], default="td3")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--hidden-actor", type=int, nargs="+", default=None)
    parser.add_argument("--hidden-critic", type=int, nargs="+", default=None)
    args = parser.parse_args(agent_args)

    print(f"\nInitializing TD3 agent...")
    print(f"  Model: {args.model_path}")
    print(f"  Actor hidden: {args.hidden_actor}")
    print(f"  Critic hidden: {args.hidden_critic}")

    return TD3HockeyAgent(
        model_path=args.model_path,
        hidden_actor=args.hidden_actor,
        hidden_critic=args.hidden_critic
    )


def main() -> None:
    print("\n" + "=" * 60)
    print("TD3 TOURNAMENT CLIENT (Clean Version)")
    print("=" * 60)
    print(f"Server: {os.environ.get('COMPRL_SERVER_URL', 'NOT SET')}")
    print(f"Port: {os.environ.get('COMPRL_SERVER_PORT', 'NOT SET')}")
    print(f"Token: {'SET' if os.environ.get('COMPRL_ACCESS_TOKEN') else 'NOT SET'}")
    print("=" * 60 + "\n")

    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
