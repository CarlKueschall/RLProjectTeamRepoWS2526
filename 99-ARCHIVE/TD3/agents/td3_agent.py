"""
AI Usage Declaration:
This file was developed with assistance from AI autocomplete features in Cursor AI IDE.
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import memory as mem
from .model import Model
from .device import get_device # put device in separate file for clarity
from .noise import GaussianNoise  # TD3 uses Gaussian noise, not OU noise (DDPG)


class QFunction(Model):
    #########################################################
    # Critic network for TD3 (Q-function)
    # Added clipping to gradients and Q-values to prevent explosion
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

        # TD3 paper: "Critic Regularization: None" - no weight decay
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)
        # SmoothL1Loss was recommended in the TD3 paper
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets, regularization=0.0):
        #########################################################
        #
        self.train()
        self.optimizer.zero_grad()

        pred = self.Q_value(observations, actions)
        loss = self.loss(pred, targets)

        if isinstance(regularization, torch.Tensor) and regularization.requires_grad:
            loss = loss + regularization

        loss.backward()

        # CRITICAL FIX: Compute gradient norm BEFORE clipping for monitoring
        grad_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        if hasattr(self, '_grad_clip') and self._grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self._grad_clip)

        self.optimizer.step()
        return loss.item(), grad_norm

    def Q_value(self, observations, actions):
        x = torch.cat([observations, actions], dim=1)
        #the CRITIC HAS TO SEE BOTH THE STATE AND THE ACTION, TO EVALUATE THE Q-VALUE.
        q_val = self.forward(x)
        
        #using soft clipping to maximize the stability, we have too many issues with exploding q-values.
        if self._q_clip is not None and self._q_clip > 0:
            if self._q_clip_mode == 'soft':
                q_val = self._q_clip * torch.tanh(q_val / self._q_clip)
            else:
                q_val = torch.clamp(q_val, -self._q_clip, self._q_clip)
        
        return q_val



class TD3Agent:
    #########################################################
    # This TD3 Agent implementation reflects the original paper implementation.
    ############################################################################
    # 
    #########################################################
    def __init__(self, observation_space, action_space, **userconfig):
        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]  # Policy output dimension (4D: agent's own actions)
        
        # Critic needs to see full 8D actions (4D agent + 4D opponent) for 26D total input
        # This matches the old version where critic had 26D input (18 obs + 8 actions)
        # Allow override for loading old checkpoints that used 4D actions (22D total: 18 obs + 4 actions)
        self._critic_action_dim = userconfig.get("critic_action_dim", 8)  # Default: 8D (new), can be 4D (old checkpoints)

        force_cpu = userconfig.get("force_cpu", False) #added this to test overhead vs parallelization on MPS.
        self.device = get_device(force_cpu=force_cpu)

        # Set up config dictionary
        self._config = {}
        # VF regularization strength (anti-lazy learning)
        # Penalizes Q(passive) > Q(active) to encourage movement toward puck
        self._config["vf_reg_lambda"] = userconfig.get("vf_reg_lambda", 0.1)
        # exploration noise strength (TD3 paper: N(0, 0.1), so eps=0.1 with sigma=1.0)
        self._config["eps"] = 0.1
        # minimum exploration noise strength
        self._config["eps_min"] = 0.1
        # exploration noise decay rate (TD3 paper: no decay, constant noise)
        self._config["eps_decay"] = 1.0
        # discount factor
        self._config["discount"] = 0.99
        # buffer size
        self._config["buffer_size"] = int(1e6)
        # batch size (TD3 paper: 100)
        self._config["batch_size"] = 100
        # learning rates (TD3 paper: 10^-3 for both actor and critic)
        self._config["learning_rate_actor"] = 1e-3
        self._config["learning_rate_critic"] = 1e-3
        self._config["hidden_sizes_actor"] = [256, 256]
        self._config["hidden_sizes_critic"] = [256, 256, 128]
        # soft update coefficient (make sure to keep this low)
        self._config["tau"] = 0.005
        # frequency of policy updates
        self._config["policy_freq"] = 2
        # frequency of target network updates
        self._config["target_update_freq"] = 2
        # target noise std 
        self._config["target_noise_std"] = 0.2 # this actually ensures that we balance over-confidence vs. doubt to keep it robust. 
        # it's kind of like an exploration mechanism but at the training updating q-value step rather than at action selection.,
        # target noise clip
        self._config["target_noise_clip"] = 0.5
        self._config["use_target_net"] = True
        self._config["grad_clip"] = 1.0
        self._config["q_clip"] = 25.0 #q-value clipping
        self._config["q_clip_mode"] = "hard"  #clipping mode hard or soft both work honestly didn't see much difference in testing

        for key in userconfig:
            self._config[key] = userconfig[key]

        self._eps = self._config['eps']
        self._eps_min = self._config.get('eps_min', 0.05)
        self._eps_decay = self._config.get('eps_decay', 0.995)

        # Initialize Gaussian noise (TD3 paper uses N(0, 0.1), not OU noise)
        # Using sigma=1.0 so that eps directly controls noise std (eps=0.1 -> N(0, 0.1))
        noise_shape = (self._action_n,)
        self.action_noise = GaussianNoise(noise_shape, sigma=1.0)

        # Replay buffer
        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        # Getting the action space bounds
        high = torch.from_numpy(self._action_space.high).to(self.device)
        low = torch.from_numpy(self._action_space.low).to(self.device)
        #########################################################

        #########################################################
        def output_activation(x):
            #########################################################
            # this is the output activation function for the actor network.
            #########################################################
            # we have to go from -1 to 1 to the action bounds which are dynamically retrieved frmo the action space above.
            #########################################################   
            # Step 1: Apply tanh to network output
            # tanh always outputs in range [-1, 1]
            tanh_x = torch.tanh(x)
            # tanh_x ∈ [-1, 1]
            # Step 2: Shift from [-1, 1] to [0, 2]
            # Add 1 so: -1 → 0, 0 → 1, 1 → 2
            added = tanh_x + 1
            # added ∈ [0, 2]
            # Step 3: Normalize to [0, 1]
            # Divide by 2: so 0 → 0, 1 → 0.5, 2 → 1
            half = added / 2
            # half ∈ [0, 1]
            # Step 4: Scale to action space width
            # Get the size of the action range
            # Example: if low=-1, high=1 → diff=2
            #          if low=-2, high=3 → diff=5
            diff = high - low
            # Multiply [0,1] by the width
            # [0, 1] × 5 = [0, 5]
            scaled = half * diff
            # scaled ∈ [0, diff]
            # Step 5: Shift to correct lower bound
            # Add the minimum value to shift from [0, diff] to [low, high]
            # Example: [0, 5] + (-2) = [-2, 3]
            result = scaled + low
            # result ∈ [low, high]
            return result 
        #########################################################
        # building the policy network
        self.policy = Model(
            input_size=self._obs_dim,
            hidden_sizes=self._config["hidden_sizes_actor"],
            output_size=self._action_n,
            activation_fun=torch.nn.ReLU(),
            output_activation=output_activation
        )
        self.policy = self.policy.to(self.device)
        #########################################################
        # building the target policy network same thing of course
        self.policy_target = Model(
            input_size=self._obs_dim,
            hidden_sizes=self._config["hidden_sizes_actor"],
            output_size=self._action_n,
            activation_fun=torch.nn.ReLU(),
            output_activation=output_activation
        )
        self.policy_target = self.policy_target.to(self.device)

        #########################################################
        # building the critic networks
        q_clip = None
        if "q_clip" in self._config:
            q_clip = self._config["q_clip"]
        q_clip_mode = "hard"
        if "q_clip_mode" in self._config:
            q_clip_mode = self._config["q_clip_mode"]
        #########################################################
        # for td3 specifically we have to use the twin critics to keep it stable.
        # this way by taking the minimum of the two critics we can ensure that we're not over-estimating the q-value
        # Critic uses 8D actions (agent + opponent) for 26D total input (18 obs + 8 actions)
        #########################################################
        self.Q1 = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._critic_action_dim,  # 8D: full action space including opponent
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=self._config["learning_rate_critic"],
            device=self.device,
            grad_clip=self._config.get("grad_clip", 1.0),
            q_clip=q_clip,
            q_clip_mode=q_clip_mode
        )

        self.Q2 = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._critic_action_dim,  # 8D: full action space including opponent
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=self._config["learning_rate_critic"],
            device=self.device,
            grad_clip=self._config.get("grad_clip", 1.0),
            q_clip=q_clip,
            q_clip_mode=q_clip_mode
        )
        # then replicate the targets equivalently
        self.Q1_target = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._critic_action_dim,  # 8D: full action space including opponent
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=0,
            device=self.device,
            q_clip=q_clip,
            q_clip_mode=q_clip_mode
        )

        self.Q2_target = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._critic_action_dim,  # 8D: full action space including opponent
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=0,
            device=self.device,
            q_clip=q_clip,
            q_clip_mode=q_clip_mode
        )

        #########################################################
        # copy the networks to the targets
        self._copy_nets()
        #########################################################

        #########################################################
        # building the optimizer (TD3 paper uses standard Adam, no weight decay)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self._config["learning_rate_actor"]
        )


        ########################################################      
        #########################################################


        self.train_iter = 0
        self.total_steps = 0

    def _copy_nets(self):
        #########################################################
        # copy the networks to the targets that's all
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def _soft_update_targets(self):
        #########################################################
        # soft update the targets to keep them from moving too fast
        #########################################################
        # tau is the update rate for the targets
        #########################################################
         # Example of what's going on: target_param = 0.005 * new_weight + 0.995 * old_target_weight
         # take a littleb it of the new, keep most of the old and merge the two 'opinions' together.
         # the naem is polyak averaging.
        tau = self._config.get("tau", 0.005)
        # this is the formula from the TD3 paper a weighted average of the current and target parameters.
        for target_param, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def act(self, observation, eps=None):
        # Convert observation to tensor
        obs_tensor = torch.from_numpy(observation.astype(np.float32)).to(self.device)
        #NOTE: NO GRADIENTS HERE!, we're just acting. in train method the policy network gets updated
        with torch.no_grad():

            action = self.policy(obs_tensor).cpu().numpy() #forward pass
        current_eps = eps if eps is not None else self._eps
        if current_eps > 0:
            noise = self.action_noise() # get the noise here for exploration
            action = action + noise * current_eps
        action = np.clip(action, self._action_space.low, self._action_space.high)
        return action
    #########################################################
    #HELPER FUNCTIONS
    def store_transition(self, transition):
        self.buffer.add_transition(transition)
    def state(self):
        # Save agent state (Q networks and policy)
        return (
            self.Q1.state_dict(),
            self.Q2.state_dict(),
            self.policy.state_dict()
        )

    def restore_state(self, state):
        self.Q1.load_state_dict(state[0])
        self.Q2.load_state_dict(state[1])
        self.policy.load_state_dict(state[2])
        self._copy_nets()
    def reset(self):
        self.action_noise.reset()
    def decay_epsilon(self):
        self._eps = max(self._eps_min, self._eps * self._eps_decay)
    def get_epsilon(self):
        return self._eps

    # END HELPER FUNCTIONS
    #########################################################
    # TRAINING LOOP
    def train(self, iter_fit=32):
        #########################################################
        # training loop
        # CORE STEPS:
        # 1. Sample batch of data from the replay buffer
        # 2. Compute the target Q-values
        # 3. Compute the loss for the Q-values
        # 4. Update the Q-values (critic networks)
        # 5. Update the policy (actor network)
        # 6. Update the targets (conditionally)
        # 7. Return the losses
        #########################################################
        losses = []
        grad_norms = []  # Track gradient norms for logging
        vf_reg_metrics_all = []  # Track VF regularization metrics
        self.train_iter += 1

        for _ in range(iter_fit):
            self.total_steps += 1

            # Sample batch from replay buffer
            data = self.buffer.sample(batch=self._config['batch_size'])

            # Split into states, actions, rewards, next states, and dones
            # Note: stored actions are 8D (4D agent + 4D opponent) for critic to see full game state
            s = torch.from_numpy(np.stack(data[:, 0]).astype(np.float32)).to(self.device)
            s_prime = torch.from_numpy(np.stack(data[:, 3]).astype(np.float32)).to(self.device)
            a_full = torch.from_numpy(np.stack(data[:, 1]).astype(np.float32)).to(self.device)  # 8D: [agent_action, opponent_action]
            a_agent = a_full[:, :4]  # Extract agent's 4D action for policy updates
            rew = torch.from_numpy(np.stack(data[:, 2]).astype(np.float32)[:, None]).to(self.device)
            done = torch.from_numpy(np.stack(data[:, 4]).astype(np.float32)[:, None]).to(self.device)

            # now we need to compute the target Q-values
            with torch.no_grad(): # NO GRAD, just outputting what the Target network currently 'thinks' about the next state.
                # Policy outputs 4D action (agent's own action)
                a_next_agent = self.policy_target(s_prime)
                noise = torch.randn_like(a_next_agent) * self._config["target_noise_std"]
                noise = noise.clamp(-self._config["target_noise_clip"], self._config["target_noise_clip"])
                a_next_agent_smooth = a_next_agent + noise
                a_next_agent_smooth = a_next_agent_smooth.clamp(
                    torch.from_numpy(self._action_space.low).to(self.device),
                    torch.from_numpy(self._action_space.high).to(self.device)
                )
                # For target Q-value, we need 8D action (agent + opponent)
                # Use opponent action from next state (extract from stored action if available, or use zeros)
                # Since we don't have opponent's next action, we'll use the opponent action from current state
                # This is an approximation but necessary for the critic architecture
                a_next_opponent = a_full[:, 4:]  # Use opponent action from current state
                a_next_full = torch.cat([a_next_agent_smooth, a_next_opponent], dim=1)  # 8D: [agent, opponent]
                
                q1_target_next = self.Q1_target.Q_value(s_prime, a_next_full)
                q2_target_next = self.Q2_target.Q_value(s_prime, a_next_full)
                q_target_next = torch.min(q1_target_next, q2_target_next)
                target_q = rew + self._config['discount'] * q_target_next * (1 - done)

                q_clip = self._config.get('q_clip', 100.0)
                q_clip_mode = self._config.get('q_clip_mode', 'soft')

                # clip the target Q-values to the range
                if q_clip_mode == 'soft':
                    target_q = q_clip * torch.tanh(target_q / q_clip)
                else:
                    target_q = torch.clamp(target_q, -q_clip, q_clip)

            #########################################################
            # VALUE FUNCTION REGULARIZATION (Anti-Lazy Learning)
            # Penalizes Q(passive) > Q(active) to prevent lazy/passive agents
            #########################################################
            vf_reg_lambda = self._config.get("vf_reg_lambda", 0.1)
            vf_reg_q1 = torch.tensor(0.0, device=self.device, requires_grad=True)
            vf_reg_q2 = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Metrics for tracking VF reg effectiveness
            vf_reg_metrics = {
                'active_ratio': 0.0,        # % of batch states where agent should be active
                'violation_ratio': 0.0,     # % of active states where Q(passive) > Q(active) (bad)
                'q_advantage_mean': 0.0,    # Mean Q(active) - Q(passive) (positive = good)
                'reg_loss': 0.0,            # Actual regularization loss added
            }

            if vf_reg_lambda > 0:
                # Compute once and reuse for both critics
                should_be_active = self._should_be_active(s)
                n_active = should_be_active.sum().item()
                vf_reg_metrics['active_ratio'] = n_active / s.shape[0]

                if n_active > 0:
                    # Construct passive vs active actions for comparison
                    a_passive_agent = torch.zeros((n_active, 4), device=self.device)
                    a_active_agent = self._generate_active_action(s[should_be_active])
                    a_opponent_reg = a_full[should_be_active, 4:]

                    a_passive_full = torch.cat([a_passive_agent, a_opponent_reg], dim=1)
                    a_active_full = torch.cat([a_active_agent, a_opponent_reg], dim=1)

                    # Q1 regularization
                    q1_passive = self.Q1.Q_value(s[should_be_active], a_passive_full)
                    q1_active = self.Q1.Q_value(s[should_be_active], a_active_full)
                    q1_advantage = q1_active - q1_passive  # Positive = active is better (good!)
                    q1_violation = torch.relu(-q1_advantage)  # Penalty when passive > active
                    vf_reg_q1 = vf_reg_lambda * q1_violation.mean()

                    # Q2 regularization
                    q2_passive = self.Q2.Q_value(s[should_be_active], a_passive_full)
                    q2_active = self.Q2.Q_value(s[should_be_active], a_active_full)
                    q2_advantage = q2_active - q2_passive
                    q2_violation = torch.relu(-q2_advantage)
                    vf_reg_q2 = vf_reg_lambda * q2_violation.mean()

                    # Collect metrics (average across both critics)
                    with torch.no_grad():
                        avg_advantage = ((q1_advantage + q2_advantage) / 2).mean().item()
                        # Count violations: states where passive > active
                        violations = ((q1_advantage < 0).float() + (q2_advantage < 0).float()) / 2
                        violation_ratio = violations.mean().item()
                        avg_reg_loss = (vf_reg_q1.item() + vf_reg_q2.item()) / 2

                        vf_reg_metrics['q_advantage_mean'] = avg_advantage
                        vf_reg_metrics['violation_ratio'] = violation_ratio
                        vf_reg_metrics['reg_loss'] = avg_reg_loss

            # Fit Q1 with regularization
            q1_loss_value, q1_grad_norm = self.Q1.fit(s, a_full, target_q, regularization=vf_reg_q1)

            # Detach target_q to avoid backward through graph twice
            target_q_detached = target_q.detach() if isinstance(target_q, torch.Tensor) else target_q
            # Fit Q2 with regularization
            q2_loss_value, q2_grad_norm = self.Q2.fit(s, a_full, target_q_detached, regularization=vf_reg_q2)
            avg_critic_loss = (q1_loss_value + q2_loss_value) / 2
            avg_critic_grad_norm = (q1_grad_norm + q2_grad_norm) / 2

            if self.total_steps % self._config["policy_freq"] == 0:
                #########################################################
                # update the policy network at the given frequency
                #########################################################
                self.optimizer.zero_grad()

                # Ask the policy right now : "what action should we take here?"
                a_current_agent = self.policy(s)  # Policy outputs 4D (agent's own action)
                # For critic evaluation, we need 8D action (agent + opponent)
                # Use opponent actions from stored buffer
                a_current_opponent = a_full[:, 4:]  # Opponent actions (4D)
                a_current_full = torch.cat([a_current_agent, a_current_opponent], dim=1)  # 8D: [agent, opponent]
                
                # Using twin critics again
                q1_val = self.Q1.Q_value(s, a_current_full)
                q2_val = self.Q2.Q_value(s, a_current_full)

                # Pick the LOWER estimate (be pessimistic, don't trust overestimated values)
                q_val = torch.min(q1_val, q2_val)

                #  we want to maximize Q-value, but optimizers minimize loss
                # Have to negate it: minimizing (-Q) = maximizing Q
                actor_loss = -q_val.mean()
                actor_loss.backward()

                # CRITICAL FIX: Compute actor gradient norm BEFORE clipping for monitoring
                actor_grad_norm = 0.0
                for p in self.policy.parameters():
                    if p.grad is not None:
                        actor_grad_norm += p.grad.data.norm(2).item() ** 2
                actor_grad_norm = actor_grad_norm ** 0.5

                if "grad_clip" in self._config:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self._config["grad_clip"])

                # Actually update the policy weights using the computed gradients
                self.optimizer.step()
                # for metrics
                losses.append((avg_critic_loss, actor_loss.detach().cpu().item()))
                grad_norms.append((avg_critic_grad_norm, actor_grad_norm))
            else:
                # in this case just log 0 for actor loss (we didn't update it)
                losses.append((avg_critic_loss, 0.0))
                grad_norms.append((avg_critic_grad_norm, 0.0))

            # separately, update the target networks (slow-moving copies for stability)
            if self.total_steps % self._config["target_update_freq"] == 0:
                # Check if we're even using target networks (some configs might disable them)
                if self._config["use_target_net"]:
                    # a soft update only as discussed above. blend old targets with new network weights
                    self._soft_update_targets()

            # Track VF reg metrics for this iteration
            vf_reg_metrics_all.append(vf_reg_metrics)

        # Aggregate VF reg metrics across all iterations
        vf_reg_summary = {
            'active_ratio': np.mean([m['active_ratio'] for m in vf_reg_metrics_all]),
            'violation_ratio': np.mean([m['violation_ratio'] for m in vf_reg_metrics_all]),
            'q_advantage_mean': np.mean([m['q_advantage_mean'] for m in vf_reg_metrics_all]),
            'reg_loss': np.mean([m['reg_loss'] for m in vf_reg_metrics_all]),
        }

        return losses, grad_norms, vf_reg_summary
    



    #########################################################
    # FUNCTIONS TO DISCOURAGE LAZY BEAHVIOR OR (PASSIVE LEARNING) IN THE AGENT.
    # The idea is to add a regularization term to the loss which is high when the agent is not actively pursuing the puck.
    # we ensure that when the agent is already active, this signal does not interfere too much with the learning process.
    def _should_be_active(self, states):
        #########################################################
        # this function checks if the agent should be active based on the state of the game.
        #########################################################
        # puck_x is the x-coordinate of the puck, if it's close to the agent, the agent should be active.
        # p1_keep_time is the time the agent has been keeping the puck, if it's been keeping the puck for a while, the agent should be active.
        #########################################################
        puck_x = states[:, 12]
        p1_keep_time = states[:, 16]
        should_be_active = (puck_x < 2.0) | (p1_keep_time > 0)
        return should_be_active
    #########################################################
    def _generate_active_action(self, states):
        #########################################################
        # Generate an "active" action for comparison with passive (zero) action.
        #
        # KEY FIX: When agent is CLOSE to puck, "active" means "hit it hard"
        # (high magnitude action), not just "move toward it" (which would be ~zero).
        #
        # When FAR from puck: active = move toward puck
        # When CLOSE to puck: active = high magnitude action (toward opponent goal)
        #########################################################
        p1_x = states[:, 0]
        p1_y = states[:, 1]
        puck_x = states[:, 12]
        puck_y = states[:, 13]

        dx = puck_x - p1_x
        dy = puck_y - p1_y
        dist = torch.sqrt(dx**2 + dy**2 + 1e-6)

        # Threshold: if agent is within 0.5 units of puck, use "hit" action instead
        CLOSE_TO_PUCK_THRESHOLD = 0.5
        is_close = dist < CLOSE_TO_PUCK_THRESHOLD

        actions = torch.zeros((states.shape[0], self._action_n), device=states.device)

        # For states FAR from puck: move toward puck
        far_mask = ~is_close
        if far_mask.any():
            dx_norm = dx[far_mask] / dist[far_mask]
            dy_norm = dy[far_mask] / dist[far_mask]
            actions[far_mask, 0] = dx_norm * 0.6
            actions[far_mask, 1] = dy_norm * 0.6

        # For states CLOSE to puck: high magnitude action toward opponent goal
        # This represents "hitting" the puck rather than just positioning
        if is_close.any():
            # Direction toward opponent goal (positive x direction)
            # Use high magnitude to represent an actual hit
            actions[is_close, 0] = 0.8  # Strong movement toward opponent side
            actions[is_close, 1] = 0.0  # Straight ahead
            # Also encourage shooting (action[3] > 0 = shoot)
            if self._action_n > 3:
                actions[is_close, 3] = 0.5  # Encourage shooting when close to puck

        # Clamp to action space bounds
        actions = actions.clamp(
            torch.from_numpy(self._action_space.low).to(states.device),
            torch.from_numpy(self._action_space.high).to(states.device)
        )

        return actions