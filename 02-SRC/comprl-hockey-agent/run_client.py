from __future__ import annotations

import argparse
import uuid
import sys
import os
import torch

import hockey.hockey_env as h_env
import numpy as np

# Add parent directory (02-SRC) to path so we can import TD3
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
from TD3.agents.td3_agent import TD3Agent

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
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

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


class TD3HockeyAgent(Agent):
    """A hockey agent using the TD3 reinforcement learning algorithm."""

    def __init__(self, model_path: str, hidden_actor: list = None, hidden_critic: list = None) -> None:
        super().__init__()

        # Initialize hockey environment to get spaces
        env = h_env.HockeyEnv()

        # Load checkpoint first to infer hidden sizes if not provided
        if model_path:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'agent_state' in checkpoint:
                agent_state = checkpoint['agent_state']
            else:
                agent_state = checkpoint

            # Infer hidden sizes from checkpoint if not provided
            if isinstance(agent_state, tuple) and len(agent_state) >= 3:
                if hidden_actor is None:
                    # Infer actor hidden sizes from policy state (state[2])
                    policy_state = agent_state[2]
                    hidden_actor = self._infer_hidden_sizes_from_state(policy_state)

                if hidden_critic is None:
                    # Infer critic hidden sizes from Q1 state (state[0])
                    q1_state = agent_state[0]
                    hidden_critic = self._infer_hidden_sizes_from_state(q1_state)

        # Use provided or inferred hidden sizes, or defaults
        if hidden_actor is None:
            hidden_actor = [256, 256]
        if hidden_critic is None:
            hidden_critic = [256, 256, 128]

        # Fix observation/action space to match checkpoint dimensions
        # Checkpoint expects: obs_dim=18, action_dim=4 (for QFunction input=22)
        from gymnasium import spaces
        obs_space = env.observation_space
        if obs_space.shape[0] > 18:
            obs_space = spaces.Box(low=obs_space.low[:18], high=obs_space.high[:18], dtype=obs_space.dtype)

        # Agent outputs 4-dim actions (player's own actions only)
        action_space = spaces.Box(
            low=env.action_space.low[:4],
            high=env.action_space.high[:4],
            dtype=env.action_space.dtype
        )

        # Create the trained TD3 agent with correct hidden sizes and action space
        self.td3_agent = TD3Agent(
            observation_space=obs_space,
            action_space=action_space,
            hidden_sizes_actor=hidden_actor,
            hidden_sizes_critic=hidden_critic
        )

        # Load trained weights from checkpoint
        if model_path:
            # Handle both absolute and relative paths
            if not os.path.isabs(model_path):
                model_path = os.path.join(script_dir, model_path)

            model_path = os.path.abspath(model_path)
            print(f"Loading TD3 model from: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            # Handle checkpoint format: if it's a dict with 'agent_state', use that
            if isinstance(checkpoint, dict) and 'agent_state' in checkpoint:
                state = checkpoint['agent_state']
            else:
                state = checkpoint

            # Fix state dict keys for compatibility with different model versions
            state = self._fix_state_dict_keys(state)

            self.td3_agent.restore_state(state)
            print("TD3 model loaded successfully")

        env.close()

    def get_step(self, observation: list[float]) -> list[float]:
        # Convert list to numpy array
        obs_array = np.array(observation, dtype=np.float32)

        # DIAGNOSTIC: Comprehensive observation analysis
        if not hasattr(self, '_obs_diagnostic_done'):
            print(f"\n{'='*70}")
            print(f"TOURNAMENT AGENT DIAGNOSTIC: First Observation Analysis")
            print(f"{'='*70}")
            print(f"Observation dimensions: {len(obs_array)}")
            print(f"\nRaw observation values:")
            print(f"  Full obs: {obs_array}")

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

            print(f"\nPerspective Detection:")
            # Check if this looks like Player 1 or Player 2 (mirrored) perspective
            player1_x = obs_array[0]
            player2_x = obs_array[6]
            puck_x = obs_array[12]

            print(f"  Player 1 X position: {player1_x:.4f}")
            print(f"  Player 2 X position: {player2_x:.4f}")
            print(f"  Puck X position: {puck_x:.4f}")

            # In normal game: Player 1 on left (negative X), Player 2 on right (positive X)
            # If mirrored: positions would be flipped
            if player1_x < 0:
                print(f"  ✓ Looks like NORMAL perspective (Player 1 on left)")
                self._player_perspective = "NORMAL"
            elif player1_x > 0:
                print(f"  ⚠ Looks like MIRRORED perspective (Player 1 on right - unusual!)")
                self._player_perspective = "MIRRORED"
            else:
                print(f"  ? Centered - cannot determine perspective")
                self._player_perspective = "UNKNOWN"

            print(f"  Inferred: Agent is playing as {'Player 1' if self._player_perspective == 'NORMAL' else 'Player 2 (mirrored)'}")

            print(f"\nObservation Statistics:")
            print(f"  Min: {np.min(obs_array):.4f}")
            print(f"  Max: {np.max(obs_array):.4f}")
            print(f"  Mean: {np.mean(obs_array):.4f}")
            print(f"  Std: {np.std(obs_array):.4f}")
            print(f"{'='*70}\n")

            self._obs_diagnostic_done = True
            self._sample_count = 0
            self._obs_samples = []

        # Sample observations for later analysis
        if not hasattr(self, '_sample_count'):
            self._sample_count = 0
            self._obs_samples = []

        self._sample_count += 1
        if self._sample_count <= 100:  # Collect first 100 samples
            self._obs_samples.append(obs_array.copy())

        # Truncate observation to 18 dimensions to match checkpoint's expected input
        if len(obs_array) > 18:
            obs_array = obs_array[:18]

        # Get action from TD3 agent (eps=0.0 disables exploration noise during inference)
        action = self.td3_agent.act(obs_array, eps=0.0)

        # Hockey environment expects 4 actions, but agent may output more
        # (e.g., if trained for self-play with dual agents)
        # Take only the first 4 actions for our player
        if len(action) > 4:
            action = action[:4]

        # Ensure action is in valid range [-1, 1] (hockey env expects normalized actions)
        action = np.clip(action, -1.0, 1.0)

        # Validate action
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f"Warning: Invalid action (NaN/Inf) detected, replacing with zeros")
            action = np.zeros(4)

        action_list = action.tolist()
        return action_list

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id, byteorder='big'))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

    @staticmethod
    def _infer_hidden_sizes_from_state(state_dict):
        """Infer hidden layer sizes from a state dict by looking at layer shapes."""
        hidden_sizes = []
        layer_idx = 0
        while f'layers.{layer_idx}.weight' in state_dict:
            weight_shape = state_dict[f'layers.{layer_idx}.weight'].shape
            hidden_sizes.append(weight_shape[0])
            layer_idx += 1

        return hidden_sizes if hidden_sizes else [256, 256]

    @staticmethod
    def _fix_state_dict_keys(state):
        """Fix state dict keys for compatibility with different model versions.

        Handles renaming of output layer from 'readout' to 'output_layer'.
        """
        if isinstance(state, tuple):
            # State is a tuple of (Q1, Q2, policy)
            fixed_state = []
            for state_dict in state:
                fixed_dict = {}
                for key, value in state_dict.items():
                    # Rename readout to output_layer
                    new_key = key.replace('readout.', 'output_layer.')
                    fixed_dict[new_key] = value
                fixed_state.append(fixed_dict)
            return tuple(fixed_state)
        else:
            # Single state dict
            fixed_dict = {}
            for key, value in state.items():
                new_key = key.replace('readout.', 'output_layer.')
                fixed_dict[new_key] = value
            return fixed_dict


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "td3"],
        default="weak",
        help="Which agent to use.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained TD3 model checkpoint",
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "td3":
        agent = TD3HockeyAgent(model_path=args.model_path)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
