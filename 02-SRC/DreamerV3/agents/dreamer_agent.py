"""
Minimal DreamerV3 Agent for Hockey.

This is a simplified implementation focused on the hockey environment.
Key components:
- RSSM world model (recurrent state space model)
- Actor-critic trained in imagination
- Sequence replay buffer

Based on:
- DreamerV3 paper (Hafner et al., 2023)
- Robot Air Hockey Challenge 2023 (Orsula et al., 2024)

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from typing import Dict, Optional, Tuple


def build_mlp(input_dim: int, output_dim: int, hidden_dims: list, activation=nn.ELU):
    """Build a simple MLP."""
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class RSSM(nn.Module):
    """
    Recurrent State Space Model - the core of DreamerV3.

    State = (h, z) where:
    - h: deterministic recurrent state (GRU hidden)
    - z: stochastic latent state

    The model learns:
    - Dynamics: predict next state from current state + action
    - Representation: encode observation into latent state
    - Reconstruction: decode latent state back to observation
    - Reward prediction: predict reward from latent state
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        stoch_size: int = 32,
        deter_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.hidden_size = hidden_size

        # Encoder: obs -> embedded
        self.encoder = build_mlp(obs_dim, hidden_size, [hidden_size] * num_layers)

        # Sequence model (GRU): (h, z, a) -> h'
        self.gru = nn.GRUCell(stoch_size + action_dim, deter_size)

        # Representation model (posterior): (h, embedded) -> z
        # Outputs mean and std for Gaussian
        self.posterior_net = build_mlp(
            deter_size + hidden_size,
            stoch_size * 2,
            [hidden_size]
        )

        # Transition model (prior): h -> z
        # Outputs mean and std for Gaussian
        self.prior_net = build_mlp(
            deter_size,
            stoch_size * 2,
            [hidden_size]
        )

        # Decoder: (h, z) -> obs
        self.decoder = build_mlp(
            deter_size + stoch_size,
            obs_dim,
            [hidden_size] * num_layers
        )

        # Reward predictor: (h, z) -> reward
        self.reward_pred = build_mlp(
            deter_size + stoch_size,
            1,
            [hidden_size]
        )

        # Continue predictor: (h, z) -> continue probability
        self.continue_pred = build_mlp(
            deter_size + stoch_size,
            1,
            [hidden_size]
        )

    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial state (h, z)."""
        h = torch.zeros(batch_size, self.deter_size, device=device)
        z = torch.zeros(batch_size, self.stoch_size, device=device)
        return h, z

    def get_feat(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Concatenate h and z to get full state features."""
        return torch.cat([h, z], dim=-1)

    def _sample_stoch(self, stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample stochastic state from mean/std."""
        mean, std = stats.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1  # Ensure positive std
        dist = D.Normal(mean, std)
        z = dist.rsample()  # Reparameterized sample
        return z, mean, std

    def observe(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Process one step with observation (for training).

        Returns:
            h_next: Next deterministic state
            z_next: Next stochastic state (from posterior)
            stats: Dictionary with prior/posterior statistics
        """
        # Encode observation
        embedded = self.encoder(obs)

        # Update deterministic state
        h_next = self.gru(torch.cat([z, action], dim=-1), h)

        # Get posterior (uses observation)
        post_stats = self.posterior_net(torch.cat([h_next, embedded], dim=-1))
        z_post, post_mean, post_std = self._sample_stoch(post_stats)

        # Get prior (doesn't use observation)
        prior_stats = self.prior_net(h_next)
        _, prior_mean, prior_std = self._sample_stoch(prior_stats)

        stats = {
            'post_mean': post_mean,
            'post_std': post_std,
            'prior_mean': prior_mean,
            'prior_std': prior_std,
        }

        return h_next, z_post, stats

    def imagine(
        self,
        action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Imagine one step without observation (for actor-critic training).

        Returns:
            h_next: Next deterministic state
            z_next: Next stochastic state (from prior)
        """
        # Update deterministic state
        h_next = self.gru(torch.cat([z, action], dim=-1), h)

        # Get prior (no observation available in imagination)
        prior_stats = self.prior_net(h_next)
        z_prior, _, _ = self._sample_stoch(prior_stats)

        return h_next, z_prior

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode latent state to observation."""
        feat = self.get_feat(h, z)
        return self.decoder(feat)

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict reward from latent state."""
        feat = self.get_feat(h, z)
        return self.reward_pred(feat).squeeze(-1)

    def predict_continue(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict continue probability from latent state."""
        feat = self.get_feat(h, z)
        return torch.sigmoid(self.continue_pred(feat)).squeeze(-1)


class Actor(nn.Module):
    """Actor network for DreamerV3."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = build_mlp(state_dim, action_dim * 2, [hidden_size, hidden_size])
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor) -> D.Distribution:
        """Get action distribution from state."""
        out = self.net(state)
        mean, std = out.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        return D.Normal(mean, std)

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action from state."""
        dist = self.forward(state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        return torch.tanh(action)  # Squash to [-1, 1]


class Critic(nn.Module):
    """Critic network for DreamerV3."""

    def __init__(self, state_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = build_mlp(state_dim, 1, [hidden_size, hidden_size])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get value from state."""
        return self.net(state).squeeze(-1)


class DreamerV3Agent(nn.Module):
    """
    Complete DreamerV3 Agent.

    Training involves:
    1. World model training on real sequences (representation learning)
    2. Actor-critic training in imagination (behavior learning)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        stoch_size: int = 32,
        deter_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        imagination_horizon: int = 50,
        gamma: float = 0.997,
        lambda_gae: float = 0.95,
        lr_world: float = 3e-4,
        lr_actor: float = 3e-5,
        lr_critic: float = 3e-5,
        free_nats: float = 1.0,
        kl_balance: float = 0.8,
        grad_clip: float = 100.0,
        device: str = 'cpu',
        **kwargs  # Accept extra kwargs for compatibility
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.imagination_horizon = imagination_horizon
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.free_nats = free_nats
        self.kl_balance = kl_balance
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        # World model
        self.world_model = RSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            stoch_size=stoch_size,
            deter_size=deter_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).to(self.device)

        # Actor and critic
        state_dim = deter_size + stoch_size
        self.actor = Actor(state_dim, action_dim, hidden_size).to(self.device)
        self.critic = Critic(state_dim, hidden_size).to(self.device)
        self.critic_target = Critic(state_dim, hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.world_opt = torch.optim.Adam(self.world_model.parameters(), lr=lr_world)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # State for acting
        self._h = None
        self._z = None

        # Training stats
        self.train_steps = 0

    def reset(self):
        """Reset agent state for new episode."""
        self._h = None
        self._z = None

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select action given observation.

        Args:
            obs: Observation array (obs_dim,)
            explore: Whether to add exploration noise

        Returns:
            action: Action array (action_dim,)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Initialize state if needed
        if self._h is None:
            self._h, self._z = self.world_model.initial_state(1, self.device)

        # Encode observation and update state
        embedded = self.world_model.encoder(obs_t)
        post_stats = self.world_model.posterior_net(
            torch.cat([self._h, embedded], dim=-1)
        )
        self._z, _, _ = self.world_model._sample_stoch(post_stats)

        # Get action from actor
        state = self.world_model.get_feat(self._h, self._z)
        action = self.actor.sample(state, deterministic=not explore)

        # Update h for next step (will be refined with actual action taken)
        # Note: This is approximate - in training we use the actual action
        self._h = self.world_model.gru(
            torch.cat([self._z, action], dim=-1),
            self._h
        )

        return action.squeeze(0).cpu().numpy()

    def train_step(self, batch: Dict) -> Dict:
        """
        Train on a batch of sequences.

        Args:
            batch: Dictionary with:
                - obs: (batch, seq_len, obs_dim)
                - action: (batch, seq_len, action_dim)
                - reward: (batch, seq_len)
                - done: (batch, seq_len)

        Returns:
            metrics: Dictionary of training metrics
        """
        self.train_steps += 1

        # Convert to tensors
        obs = torch.FloatTensor(batch['obs']).to(self.device)
        action = torch.FloatTensor(batch['action']).to(self.device)
        reward = torch.FloatTensor(batch['reward']).to(self.device)
        done = torch.BoolTensor(batch['done']).to(self.device)

        batch_size, seq_len = obs.shape[:2]

        # ==================
        # World Model Training
        # ==================
        self.world_opt.zero_grad()

        # Initialize state
        h, z = self.world_model.initial_state(batch_size, self.device)

        # Process sequence
        recon_loss = 0.0
        reward_loss = 0.0
        kl_loss = 0.0

        for t in range(seq_len - 1):
            h, z, stats = self.world_model.observe(obs[:, t], action[:, t], h, z)

            # Reconstruction loss
            obs_pred = self.world_model.decode(h, z)
            recon_loss += F.mse_loss(obs_pred, obs[:, t + 1])

            # Reward prediction loss
            reward_pred = self.world_model.predict_reward(h, z)
            reward_loss += F.mse_loss(reward_pred, reward[:, t + 1])

            # KL divergence (posterior vs prior)
            post_dist = D.Normal(stats['post_mean'], stats['post_std'])
            prior_dist = D.Normal(stats['prior_mean'], stats['prior_std'])
            kl = D.kl_divergence(post_dist, prior_dist).sum(-1).mean()
            kl = torch.clamp(kl, min=self.free_nats)  # Free nats
            kl_loss += kl

        # Total world model loss
        world_loss = recon_loss + reward_loss + kl_loss
        world_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), self.grad_clip)
        self.world_opt.step()

        # ==================
        # Actor-Critic Training (in imagination)
        # ==================

        # Get starting states from real data (no gradients needed for starting point)
        with torch.no_grad():
            h, z = self.world_model.initial_state(batch_size, self.device)
            for t in range(min(seq_len // 2, 10)):  # Use first half to get varied states
                h, z, _ = self.world_model.observe(obs[:, t], action[:, t], h, z)
        h_start = h.detach()
        z_start = z.detach()

        # Imagine trajectories for critic training (no actor gradients needed)
        with torch.no_grad():
            h_imag = h_start
            z_imag = z_start
            states_for_critic = []
            rewards_imag = []
            continues_imag = []

            for _ in range(self.imagination_horizon):
                state = self.world_model.get_feat(h_imag, z_imag)
                states_for_critic.append(state)
                action_imag = self.actor.sample(state)
                h_imag, z_imag = self.world_model.imagine(action_imag, h_imag, z_imag)
                rewards_imag.append(self.world_model.predict_reward(h_imag, z_imag))
                continues_imag.append(self.world_model.predict_continue(h_imag, z_imag))

            states_for_critic = torch.stack(states_for_critic, dim=1)
            rewards_imag = torch.stack(rewards_imag, dim=1)
            continues_imag = torch.stack(continues_imag, dim=1)

            # Compute lambda-returns as targets
            values_target = self.critic_target(states_for_critic)
            returns = self._compute_lambda_returns(rewards_imag, values_target, continues_imag)

        # Critic loss
        self.critic_opt.zero_grad()
        value_pred = self.critic(states_for_critic[:, :-1])
        critic_loss = F.mse_loss(value_pred, returns[:, :-1])
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        # Actor loss - imagine trajectories WITH gradients through actor
        # Freeze world model during actor training
        for param in self.world_model.parameters():
            param.requires_grad = False

        self.actor_opt.zero_grad()

        h_imag = h_start
        z_imag = z_start
        actor_values = []

        for t in range(self.imagination_horizon):
            state = self.world_model.get_feat(h_imag, z_imag)

            # Get action from actor (this is where gradients flow)
            action_imag = self.actor.sample(state)

            # Imagine next state (world model is frozen, but computation graph still flows through action)
            h_imag, z_imag = self.world_model.imagine(action_imag, h_imag, z_imag)

            # Get value of resulting state
            next_state = self.world_model.get_feat(h_imag, z_imag)
            value = self.critic(next_state)
            actor_values.append(value)

        # Actor loss: maximize expected value
        actor_values = torch.stack(actor_values, dim=1)  # (batch, horizon)
        actor_loss = -actor_values.mean()

        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        # Unfreeze world model
        for param in self.world_model.parameters():
            param.requires_grad = True

        # Update target critic
        if self.train_steps % 100 == 0:
            self._update_target()

        return {
            'world_loss': world_loss.item(),
            'recon_loss': recon_loss.item() / seq_len,
            'reward_loss': reward_loss.item() / seq_len,
            'kl_loss': kl_loss.item() / seq_len,
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
        }

    def _compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        continues: torch.Tensor
    ) -> torch.Tensor:
        """Compute lambda-returns for value targets."""
        horizon = rewards.shape[1]
        returns = torch.zeros_like(rewards)

        # Bootstrap from last value
        returns[:, -1] = values[:, -1]

        for t in reversed(range(horizon - 1)):
            returns[:, t] = (
                rewards[:, t] +
                self.gamma * continues[:, t] * (
                    (1 - self.lambda_gae) * values[:, t + 1] +
                    self.lambda_gae * returns[:, t + 1]
                )
            )

        return returns

    def _update_target(self, tau: float = 0.02):
        """Soft update target critic."""
        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def state_dict(self) -> Dict:
        """Get agent state for saving."""
        return {
            'world_model': self.world_model.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'world_opt': self.world_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'train_steps': self.train_steps,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load agent state."""
        self.world_model.load_state_dict(state_dict['world_model'])
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.world_opt.load_state_dict(state_dict['world_opt'])
        self.actor_opt.load_state_dict(state_dict['actor_opt'])
        self.critic_opt.load_state_dict(state_dict['critic_opt'])
        self.train_steps = state_dict['train_steps']
