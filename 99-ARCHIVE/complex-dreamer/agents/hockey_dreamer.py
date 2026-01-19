"""
HockeyDreamer: DreamerV3 Agent for Hockey Environment.

Complete agent class that combines:
- World Model (RSSM + prediction heads)
- Behavior Model (Actor-Critic)

Provides clean interface for:
- Acting in environment
- Training on batches
- Self-play opponent management (state/restore_state)

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import gc
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from models.world_model import WorldModel
from models.behavior import Behavior
from utils.math_ops import adaptive_gradient_clip


class HockeyDreamer(nn.Module):
    """
    Complete DreamerV3 agent for hockey.

    The agent learns by:
    1. Collecting experience in the real environment
    2. Training world model to predict dynamics
    3. Training actor-critic entirely in imagination

    Interface designed for easy self-play integration.
    """

    def __init__(
        self,
        obs_dim: int = 18,
        action_dim: int = 4,
        hidden_size: int = 256,
        num_categories: int = 32,
        num_classes: int = 32,
        recurrent_size: int = 256,
        embed_dim: int = 256,
        horizon: int = 15,
        imagine_batch_size: int = 256,
        gamma: float = 0.997,
        lambda_gae: float = 0.95,
        kl_free: float = 1.0,
        kl_dyn_scale: float = 0.5,
        kl_rep_scale: float = 0.1,
        unimix: float = 0.01,
        entropy_scale: float = 3e-3,
        lr_world: float = 4e-5,
        lr_actor: float = 4e-5,
        lr_critic: float = 4e-5,
        grad_clip: float = 100.0,
        use_agc: bool = True,
        agc_clip: float = 0.3,
        terminal_reward_weight: float = 1000.0,
        device: str = 'cpu',
    ):
        """
        Args:
            obs_dim: Observation dimension (18 for hockey)
            action_dim: Action dimension (4 for hockey)
            hidden_size: MLP hidden layer size
            num_categories: Number of categorical variables for latent (default: 32)
            num_classes: Classes per categorical variable (default: 32)
            recurrent_size: RSSM deterministic state dimension
            embed_dim: Encoder output dimension
            horizon: Imagination rollout length
            imagine_batch_size: Number of starting states for imagination
            gamma: Discount factor
            lambda_gae: TD(λ) lambda parameter
            kl_free: Free bits threshold for KL loss
            kl_dyn_scale: Dynamics KL weight
            kl_rep_scale: Representation KL weight
            unimix: Uniform mixing ratio for categorical latents (default: 1%)
            entropy_scale: Entropy bonus coefficient
            lr_world: World model learning rate
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            grad_clip: Standard gradient clipping norm (used if use_agc=False)
            use_agc: Use Adaptive Gradient Clipping (default: True)
            agc_clip: AGC clip factor (default: 0.3, from DreamerV3 paper)
            terminal_reward_weight: Weight for non-zero rewards in loss (default: 1000)
            device: Torch device
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.imagine_batch_size = imagine_batch_size
        self.grad_clip = grad_clip
        self.use_agc = use_agc
        self.agc_clip = agc_clip
        self.device = torch.device(device)

        # Latent size is categorical: num_categories * num_classes
        latent_size = num_categories * num_classes

        # World model with categorical latents
        self.world_model = WorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_categories=num_categories,
            num_classes=num_classes,
            recurrent_size=recurrent_size,
            embed_dim=embed_dim,
            kl_free=kl_free,
            kl_dyn_scale=kl_dyn_scale,
            kl_rep_scale=kl_rep_scale,
            unimix=unimix,
            terminal_reward_weight=terminal_reward_weight,
            device=device,
        ).to(self.device)

        # Behavior model (actor-critic)
        feature_dim = recurrent_size + latent_size
        self.behavior = Behavior(
            feature_dim=feature_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            gamma=gamma,
            lambda_gae=lambda_gae,
            entropy_scale=entropy_scale,
            device=device,
        ).to(self.device)

        # Optimizers
        self.world_opt = torch.optim.Adam(
            self.world_model.parameters(),
            lr=lr_world,
        )
        self.actor_opt = torch.optim.Adam(
            self.behavior.policy.parameters(),
            lr=lr_actor,
        )
        self.critic_opt = torch.optim.Adam(
            self.behavior.value.parameters(),
            lr=lr_critic,
        )

        # Recurrent state for acting
        self._state = None

        # Training stats
        self.train_steps = 0

    def reset(self):
        """Reset recurrent state for new episode."""
        self._state = None

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action given observation.

        Args:
            obs: Observation array (obs_dim,)
            deterministic: If True, use mode instead of sampling

        Returns:
            action: Action array (action_dim,)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Initialize state if needed
        if self._state is None:
            self._state = self.world_model.initial_state(1)

        # Encode observation
        embed = self.world_model.encode(obs_t)

        # Get previous action (zeros for first step)
        if not hasattr(self, '_last_action'):
            self._last_action = torch.zeros(1, self.action_dim, device=self.device)

        # Update state using posterior (we have the real observation)
        posterior, _ = self.world_model.dynamics.posterior_step(
            self._state,
            self._last_action,
            embed,
        )
        self._state = posterior

        # Get action from policy
        features = self.world_model.get_features(self._state)
        action = self.behavior.act(features, deterministic=deterministic)

        # Store for next step
        self._last_action = action

        return action.squeeze(0).cpu().numpy()

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        One complete training iteration.

        Args:
            batch: Dictionary with sequences:
                - obs: (batch, seq_len, obs_dim)
                - action: (batch, seq_len, action_dim)
                - reward: (batch, seq_len)
                - is_first: (batch, seq_len)
                - is_terminal: (batch, seq_len)

        Returns:
            metrics: Dictionary of training metrics
        """
        self.train_steps += 1

        # Convert to tensors
        batch_t = {
            k: torch.FloatTensor(v).to(self.device)
            for k, v in batch.items()
        }

        metrics = {}

        # === Train World Model ===
        self.world_opt.zero_grad()

        world_loss, world_metrics = self.world_model.compute_loss(batch_t)
        world_loss.backward()

        # Apply gradient clipping (AGC or standard)
        world_grad_norm = self._clip_gradients(self.world_model.parameters())
        self.world_opt.step()

        metrics.update(world_metrics)
        metrics['grad/world'] = world_grad_norm

        # === Train Actor-Critic in Imagination ===
        # Get starting states from real data (subsampled for efficiency)
        start_states = self.world_model.get_start_states(
            batch_t, num_starts=self.imagine_batch_size
        )

        # === Train Critic ===
        # Imagine trajectories without gradients (critic doesn't need policy gradients)
        with torch.no_grad():
            states, actions, rewards, continues = self.world_model.imagine(
                policy=self.behavior.policy,
                start_state=start_states,
                horizon=self.horizon,
            )

        # Store imagination stats
        imagine_reward_mean = rewards.mean().item()
        imagine_reward_std = rewards.std().item()
        imagine_continue_mean = continues.mean().item()

        # Train critic (separate method - no actor graph built)
        self.critic_opt.zero_grad()
        critic_loss, critic_metrics = self.behavior.train_critic(
            states, rewards, continues
        )
        critic_loss.backward()

        # Apply gradient clipping (AGC or standard)
        critic_grad_norm = self._clip_gradients(self.behavior.value.parameters())
        self.critic_opt.step()

        # Explicitly free critic tensors and clear graph
        del states, actions, rewards, continues, critic_loss
        self._clear_memory_cache()

        # === Train Actor ===
        # Re-imagine WITH gradients through policy for actor training
        self.actor_opt.zero_grad()
        states_actor, actions_actor, rewards_actor, continues_actor = self.world_model.imagine(
            policy=self.behavior.policy,
            start_state=start_states,
            horizon=self.horizon,
        )

        actor_loss, actor_metrics = self.behavior.train_actor(
            states_actor, actions_actor, rewards_actor, continues_actor
        )
        actor_loss.backward()

        # Apply gradient clipping (AGC or standard)
        actor_grad_norm = self._clip_gradients(self.behavior.policy.parameters())
        self.actor_opt.step()

        # Explicitly free actor tensors and clear graph
        del states_actor, actions_actor, rewards_actor, continues_actor, start_states, actor_loss
        self._clear_memory_cache()

        # Combine metrics
        metrics.update(critic_metrics)
        metrics.update(actor_metrics)
        metrics['grad/actor'] = actor_grad_norm
        metrics['grad/critic'] = critic_grad_norm
        metrics['train_steps'] = self.train_steps

        # Add imagination stats
        metrics['imagine/reward_mean'] = imagine_reward_mean
        metrics['imagine/reward_std'] = imagine_reward_std
        metrics['imagine/continue_mean'] = imagine_continue_mean

        return metrics

    def _clear_memory_cache(self):
        """Clear GPU/MPS memory cache and run garbage collection."""
        # Force Python garbage collection to release tensor references
        gc.collect()

        # Clear device memory cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            # Also synchronize to ensure all operations complete
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

    def _compute_grad_norm(self, parameters) -> float:
        """Compute total gradient norm across parameters."""
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _clip_gradients(self, parameters) -> float:
        """
        Apply gradient clipping (AGC or standard norm clipping).

        Returns:
            grad_norm: Total gradient norm before clipping
        """
        if self.use_agc:
            return adaptive_gradient_clip(parameters, clip_factor=self.agc_clip)
        else:
            grad_norm = self._compute_grad_norm(parameters)
            nn.utils.clip_grad_norm_(parameters, self.grad_clip)
            return grad_norm

    # === Self-Play Interface ===

    def state(self) -> Dict:
        """
        Serialize agent state for opponent pool.

        Returns state dict that can be saved and loaded later.
        """
        return {
            'world_model': self.world_model.state_dict(),
            'behavior': self.behavior.state_dict_custom(),
            'world_opt': self.world_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'train_steps': self.train_steps,
        }

    def restore_state(self, state: Dict):
        """
        Load agent state from opponent pool.

        Args:
            state: State dict from state() method
        """
        self.world_model.load_state_dict(state['world_model'])
        self.behavior.load_state_dict_custom(state['behavior'])
        self.world_opt.load_state_dict(state['world_opt'])
        self.actor_opt.load_state_dict(state['actor_opt'])
        self.critic_opt.load_state_dict(state['critic_opt'])
        self.train_steps = state['train_steps']

    def save(self, path: str):
        """Save agent to file."""
        torch.save({'agent_state': self.state()}, path)

    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)
        if 'agent_state' in checkpoint:
            self.restore_state(checkpoint['agent_state'])
        else:
            self.restore_state(checkpoint)

    # === Compatibility Methods ===

    def state_dict(self) -> Dict:
        """PyTorch-compatible state dict."""
        return self.state()

    def load_state_dict(self, state: Dict):
        """PyTorch-compatible state loading."""
        self.restore_state(state)
