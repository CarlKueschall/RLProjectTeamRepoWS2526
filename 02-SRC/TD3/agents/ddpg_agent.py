import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import memory as mem
from .model import Model
from .device import get_device
from .noise import OUNoise


class QFunction(Model):
    #########################################################
    # Critic Network: Q(s, a) -> Q-value
    #########################################################
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100, 100],
                 learning_rate=0.0002, device=None, grad_clip=1.0, q_clip=None, q_clip_mode='hard'):
        super().__init__(input_size=observation_dim + action_dim,
                         hidden_sizes=hidden_sizes,
                         output_size=1)

        self.device = device if device else torch.device("cpu")
        self.to(self.device)
        self._grad_clip = grad_clip
        self._q_clip = q_clip
        self._q_clip_mode = q_clip_mode

        # Phase 1 Fix: Use betas=(0.9, 0.9) instead of default (0.9, 0.999)
        # This prevents policy collapse in non-stationary environments (Dohare et al., 2023)
        # Add L2 regularization (weight_decay) to prevent weight instability
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate,
                                          eps=0.000001,
                                          betas=(0.9, 0.9),
                                          weight_decay=1e-4)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets, weights=None, regularization=0.0):
        #########################################################
        # Update critic network
        #########################################################
        self.train()
        self.optimizer.zero_grad()

        pred = self.Q_value(observations, actions)
        loss = self.loss(pred, targets)
        
        if weights is not None:
            weights_tensor = torch.from_numpy(weights).to(self.device).unsqueeze(1)
            loss = (weights_tensor * loss).mean()
        else:
            loss = loss.mean()
        
        if isinstance(regularization, torch.Tensor) and regularization.requires_grad:
            loss = loss + regularization

        loss.backward()

        if hasattr(self, '_grad_clip') and self._grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self._grad_clip)

        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        #########################################################
        # Compute Q-value for state-action pairs
        #########################################################
        x = torch.cat([observations, actions], dim=1)
        q_val = self.forward(x)
        
        # Q-value clipping for stability
        if self._q_clip is not None and self._q_clip > 0:
            if self._q_clip_mode == 'soft':
                q_val = self._q_clip * torch.tanh(q_val / self._q_clip)
            else:
                q_val = torch.clamp(q_val, -self._q_clip, self._q_clip)
        
        return q_val


class DDPGAgent:
    #########################################################
    # DDPG Agent for continuous control
    #########################################################
    # Components:
    # - Actor (Policy): mu(s) -> action
    # - Critic (Q): Q(s, a) -> Q-value
    # - Target networks for stability
    # - Replay buffer for off-policy learning
    #########################################################

    def __init__(self, observation_space, action_space, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise ValueError(f'Observation space {observation_space} incompatible '
                                   f'with {self}. (Require: Box)')
        if not isinstance(action_space, spaces.box.Box):
            raise ValueError(f'Action space {action_space} incompatible with {self}.'
                                   f' (Require Box)')

        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]

        #########################################################
        # Device setup
        #########################################################
        force_cpu = userconfig.get("force_cpu", False)
        self.device = get_device(force_cpu=force_cpu)

        #########################################################
        # Default hyperparameters (original DDPG paper)
        #########################################################
        self._config = {
            "eps": 0.3,                      # Exploration noise strength
            "eps_min": 0.3,                  # Minimum noise (high exploration prevents lazy policy)
            "eps_decay": 0.999,             # Noise decay per episode
            "discount": 0.99,                # Gamma
            "buffer_size": int(1e6),         # Replay buffer size
            "batch_size": 256,               # Batch size
            "learning_rate_actor": 3e-4,     # Actor LR
            "learning_rate_critic": 3e-4,    # Critic LR
            "hidden_sizes_actor": [256, 256], # Actor hidden sizes
            "hidden_sizes_critic": [256, 256, 128], # Critic hidden sizes
            "tau": 0.005,                    # Soft update coefficient
            "use_target_net": True,
            "grad_clip": 1.0,                # Gradient clipping norm
            "q_clip": 25.0,                  # Q-value clipping
            "q_clip_mode": "hard",           # Clipping mode
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']
        self._eps_min = self._config.get('eps_min', 0.05)
        self._eps_decay = self._config.get('eps_decay', 0.995)

        #########################################################
        # Exploration noise (CPU-based)
        #########################################################
        noise_shape = (self._action_n,)
        self.action_noise = OUNoise(noise_shape)

        #########################################################
        # Replay buffer setup (with optional dual buffers for self-play)
        #########################################################
        use_dual_buffers = self._config.get("use_dual_buffers", False)
        if use_dual_buffers:
            buffer_size_each = self._config["buffer_size"] // 2
            self.buffer_anchor = mem.Memory(max_size=buffer_size_each)
            self.buffer_pool = mem.Memory(max_size=buffer_size_each)
            self.buffer = self.buffer_anchor
            print("Dual replay buffers enabled: {} each (anchor + pool)".format(buffer_size_each))
        else:
            self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        #########################################################
        # Critic (Q-function)
        #########################################################
        self.Q = QFunction(observation_dim=self._obs_dim,
                           action_dim=self._action_n,
                           hidden_sizes=self._config["hidden_sizes_critic"],
                           learning_rate=self._config["learning_rate_critic"],
                           device=self.device,
                           grad_clip=self._config["grad_clip"],
                           q_clip=self._config.get("q_clip"),
                           q_clip_mode=self._config.get("q_clip_mode", "hard"))

        # Target critic
        self.Q_target = QFunction(observation_dim=self._obs_dim,
                                  action_dim=self._action_n,
                                  hidden_sizes=self._config["hidden_sizes_critic"],
                                  learning_rate=0,
                                  device=self.device,
                                  grad_clip=self._config["grad_clip"],
                                  q_clip=self._config.get("q_clip"),
                                  q_clip_mode=self._config.get("q_clip_mode", "hard"))

        #########################################################
        # Actor (Policy) with bounded output
        #########################################################
        # Move action bounds to device
        high = torch.from_numpy(self._action_space.high).to(self.device)
        low = torch.from_numpy(self._action_space.low).to(self.device)
        
        def output_activation(x):
            #########################################################
            # Map tanh output [-1, 1] to action space bounds
            #########################################################
            # Step 1: Apply tanh to network output
            tanh_x = torch.tanh(x)
            # tanh_x ∈ [-1, 1]
            # Step 2: Shift from [-1, 1] to [0, 2]
            added = tanh_x + 1
            # added ∈ [0, 2]
            # Step 3: Normalize to [0, 1]
            half = added / 2
            # half ∈ [0, 1]
            # Step 4: Scale to action space width
            diff = high - low
            scaled = half * diff
            # Step 5: Shift to action space bounds
            result = scaled + low
            return result

        self.policy = Model(input_size=self._obs_dim,
                            hidden_sizes=self._config["hidden_sizes_actor"],
                            output_size=self._action_n,
                            activation_fun=torch.nn.ReLU(),
                            output_activation=output_activation).to(self.device)

        self.policy_target = Model(input_size=self._obs_dim,
                                   hidden_sizes=self._config["hidden_sizes_actor"],
                                   output_size=self._action_n,
                                   activation_fun=torch.nn.ReLU(),
                                   output_activation=output_activation).to(self.device)

        self._copy_nets()

        #########################################################
        # Actor optimizer
        #########################################################
        # Phase 1 Fix: Use betas=(0.9, 0.9) instead of default (0.9, 0.999)
        # This prevents policy collapse in non-stationary environments (Dohare et al., 2023)
        # Add L2 regularization (weight_decay) to prevent weight instability
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self._config["learning_rate_actor"],
                                          eps=0.000001,
                                          betas=(0.9, 0.9),
                                          weight_decay=1e-4)

        self.train_iter = 0

    def _copy_nets(self):
        #########################################################
        # Copy main networks to target networks (hard update)
        #########################################################
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def _soft_update_targets(self):
        #########################################################
        # Soft update: target = tau * main + (1-tau) * target
        #########################################################
        tau = self._config.get("tau", 0.005)

        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def act(self, observation, eps=None):
        #########################################################
        # Select action with exploration noise
        #########################################################
        # Convert to tensor first, then to device
        obs_tensor = torch.from_numpy(observation.astype(np.float32)).to(self.device)

        with torch.no_grad():
            action = self.policy(obs_tensor).cpu().numpy()

        current_eps = eps if eps is not None else self._eps
        if current_eps > 0:
            noise = self.action_noise()
            action = action + noise * current_eps

        action = np.clip(action, self._action_space.low, self._action_space.high)
        return action

    def store_transition(self, transition):
        #########################################################
        # Store (s, a, r, s', done) in replay buffer
        #########################################################
        self.buffer.add_transition(transition)

    def state(self):
        #########################################################
        # Return network states for checkpointing
        # Returns (Q, Q, policy) to match TD3's (Q1, Q2, policy) format
        #########################################################
        q_state = self.Q.state_dict()
        return (q_state, q_state, self.policy.state_dict())

    def restore_state(self, state):
        #########################################################
        # Restore network states from checkpoint
        # Accepts (Q1, Q2, policy) but uses Q1 for both Q and Q_target
        #########################################################
        self.Q.load_state_dict(state[0])
        self.Q_target.load_state_dict(state[0])  # Use Q1 for target too
        self.policy.load_state_dict(state[2])
        self._copy_nets()

    def reset(self):
        #########################################################
        # Reset noise process
        #########################################################
        self.action_noise.reset()

    def decay_epsilon(self):
        #########################################################
        # Decay exploration noise (called after each episode)
        #########################################################
        self._eps = max(self._eps_min, self._eps * self._eps_decay)

    def get_epsilon(self):
        #########################################################
        # Get current epsilon value
        #########################################################
        return self._eps

    def train(self, iter_fit=32):
        #########################################################
        # Train agent on minibatches from replay buffer
        #########################################################
        # Uses soft target updates (Polyak averaging) like original DDPG paper
        # Returns: List of (critic_loss, actor_loss) tuples
        #########################################################
        losses = []
        self.train_iter += 1

        # Soft target updates every training step (original DDPG)
        if self._config["use_target_net"]:
            self._soft_update_targets()

        for _ in range(iter_fit):
            #########################################################
            # Sample from replay buffer (with dual buffer support)
            #########################################################
            use_dual_buffers = self._config.get("use_dual_buffers", False)
            if use_dual_buffers and hasattr(self, 'buffer_pool') and len(self.buffer_pool) > 0:
                batch_size = self._config['batch_size']
                anchor_batch_size = max(1, batch_size // 3)
                pool_batch_size = batch_size - anchor_batch_size

                # Only sample if there are enough experiences in the buffer
                if len(self.buffer_anchor) >= anchor_batch_size and len(self.buffer_pool) >= pool_batch_size:
                    data_anchor = self.buffer_anchor.sample(batch=anchor_batch_size)
                    data_pool = self.buffer_pool.sample(batch=pool_batch_size)
                    data = np.vstack([data_anchor, data_pool])
                elif len(self.buffer_anchor) >= batch_size:
                    data = self.buffer_anchor.sample(batch=batch_size)
                elif len(self.buffer_pool) >= batch_size:
                    data = self.buffer_pool.sample(batch=batch_size)
                else:
                    # If not, sample from both buffers in a balanced way
                    total_size = len(self.buffer_anchor) + len(self.buffer_pool)
                    if total_size > 0:
                        n_anchor = min(len(self.buffer_anchor), batch_size // 2)
                        n_pool = min(len(self.buffer_pool), batch_size - n_anchor)
                        if n_anchor > 0 and n_pool > 0:
                            data_anchor = self.buffer_anchor.sample(batch=n_anchor)
                            data_pool = self.buffer_pool.sample(batch=n_pool)
                            data = np.vstack([data_anchor, data_pool])
                        elif n_anchor > 0:
                            data = self.buffer_anchor.sample(batch=n_anchor)
                        else:
                            data = self.buffer_pool.sample(batch=n_pool)
                    else:
                        continue
            else:
                # If not using dual buffers, sample from main buffer
                if len(self.buffer) < self._config['batch_size']:
                    continue
                data = self.buffer.sample(batch=self._config['batch_size'])

            #########################################################
            # Convert to tensors and move to device
            #########################################################
            s = torch.from_numpy(np.stack(data[:, 0]).astype(np.float32)).to(self.device)
            a = torch.from_numpy(np.stack(data[:, 1]).astype(np.float32)).to(self.device)
            rew = torch.from_numpy(np.stack(data[:, 2]).astype(np.float32)[:, None]).to(self.device)
            s_prime = torch.from_numpy(np.stack(data[:, 3]).astype(np.float32)).to(self.device)
            done = torch.from_numpy(np.stack(data[:, 4]).astype(np.float32)[:, None]).to(self.device)

            #########################################################
            # Critic Update
            #########################################################
            with torch.no_grad():
                # Target policy action
                a_next = self.policy_target(s_prime)
                # Target Q-value
                q_target_next = self.Q_target.Q_value(s_prime, a_next)
                # Bellman target
                target_q = rew + self._config['discount'] * q_target_next * (1 - done)

            # Update critic
            q_loss_value = self.Q.fit(s, a, target_q)

            #########################################################
            # Actor Update
            #########################################################
            self.optimizer.zero_grad()

            # Current policy actions
            a_current = self.policy(s)
            q_val = self.Q.Q_value(s, a_current)
            actor_loss = -q_val.mean()

            actor_loss.backward()

            # Gradient clipping
            if "grad_clip" in self._config:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self._config["grad_clip"])

            self.optimizer.step()

            losses.append((q_loss_value, actor_loss.detach().cpu().item()))

        return losses

