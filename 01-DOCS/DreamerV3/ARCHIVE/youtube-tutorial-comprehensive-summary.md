# DreamerV3: Comprehensive Tutorial Summary

## Table of Contents
1. [Introduction & Core Concept](#introduction--core-concept)
2. [Paper & Architecture](#paper--architecture)
3. [Architecture Diagrams](#architecture-diagrams)
4. [Lambda Returns & Loss Calculations](#lambda-returns--loss-calculations)
5. [Implementation Comparison](#implementation-comparison)
6. [Code Structure Overview](#code-structure-overview)
7. [World Model Training](#world-model-training)
8. [Behavior Training](#behavior-training)
9. [Neural Network Details](#neural-network-details)
10. [Key Design Decisions](#key-design-decisions)

---

## Introduction & Core Concept

### What is DreamerV3?

DreamerV3 is one of the best reinforcement learning algorithms. The fundamental difference from classic RL:
- **Classic RL**: Agent plays the game to learn behavior directly
- **DreamerV3**: Agent plays the game to learn a **world model** (understands how the world works), then learns behavior based on this model

### Why This Approach?
1. More efficient sample usage - learns from imagination rollouts
2. Can handle sparse reward environments better through dense reward prediction
3. Learns general understanding of dynamics that transfers between tasks

---

## Paper & Architecture

### Publication Details
- **Paper**: "Mastering Diverse Domains through World Models"
- **Length**: ~11 pages (rest is references/additions)
- **Language Quality**: Surprisingly approachable and pleasant to read
- **Recommendation**: Don't be afraid of academic papers - they're written by people for people

### Abstract Key Points
- Creates a general, universal algorithm (not task-specific)
- Solves 150+ tasks
- Demonstrates capability on complex tasks like Minecraft (obtaining diamonds)
- Outperforms prior methods like DQN and MuZero

### Key Concepts to Know
- **DQN**: Deep Q-Networks - basic deep RL
- **MuZero**: Another world model approach - good to understand conceptually
- These are foundational to understanding why DreamerV3 is an improvement

---

## Architecture Diagrams

### Three-Step Training Pipeline

1. **Gather Data**: Play the game to collect real experience
2. **Train World Model**: Learn to predict what happens
3. **Train Behavior**: Learn policy using world model's predictions

### Paper's Official Diagrams - Components

#### Encoder
- **Input**: Observation (image or vector)
- **Output**: Discrete representation Z
- **Role**: Compresses sensory input into manageable form

#### Sequence Model with Recurrent State H
- **Input**: Recurrent state, discrete representation, actions
- **Output**: New recurrent state
- **Key Role**: Maintains temporal context

#### Discrete Representation
- **Format**: One-hot encoded matrix
- **Dimensions**: Typically 32×32 (variable but paper uses this)
- **Why Discrete?**: Better performance than continuous alternatives
  - See Aiden Maker's video on this topic for deep explanation
  - Long story short: performs better in practice

#### Two Ways to Create Representation (Crucial Concept)
1. **Encoder Path**: From recurrent state + real observation
   - Has access to actual observation
   - Better informed, acts as "teacher"

2. **Dynamics Predictor Path**: From recurrent state only
   - No access to real observation
   - Acts as "student" guided by encoder
   - This is what's used during imagination rollouts

**Why Both?**
- World model needs to make predictions without seeing real observations (during behavior learning)
- Encoder guides/trains the dynamics predictor
- Cooperation: teacher-student relationship

#### Reward & Continue Predictors
- **Reward Predictor**: Predicts rewards from (recurrent state + discrete representation)
- **Continue Predictor**: Predicts whether episode continues
- **Both Use**: Full state representation (recurrent + latent combined)

#### Decoder
- **Input**: Full state (recurrent + discrete representation)
- **Output**: Reconstructed observation
- **Purpose**: Applies "pressure" to keep representations informative
  - Prevents information loss in compression
  - Acts as regularization
  - The mysterious paper phrase: "inputs are reconstructed to shape the representations"
- **Note**: Reconstruction itself isn't directly useful, but the pressure it applies matters

#### Recurrent State Interpretation
- Encodes the past
- Combined with current observation to represent present state
- Dynamics predictor uses only past to predict present

#### Full State Concept
- Concatenation of: recurrent state (past) + discrete representation (present)
- Complete representation of world state at one timestep
- Foundation for reward/continue predictions
- Input for decoder

---

## Architecture Diagrams

### Presenter's Improved Diagrams

#### Split Design Philosophy
- **Encoder split into two**:
  - Allows treating observation processing as universal "choke point"
  - Standardizes different observation types (images, vectors, etc.)
  - Creates clean separation of concerns

#### Naming Changes from Paper
- **Posterior Net**: Creates posterior representation (encoder + recurrent state + observation)
  - "Better informed belief" about current state
  - Has maximum information available

- **Prior Net**: Creates prior representation (recurrent state only)
  - "What we think state is" without seeing observation
  - Used during imagination

- **Recurrent State (H)**: Deterministic state capturing history
- **Latent State (Z)**: Discrete representation of current state
  - Can be either posterior or prior version
  - People also call this: "stochastic state"

#### World Model Training Phase (WMT) Diagram

**Initialization**
- Recurrent state initialized to zeros (no past initially)
- Discrete representation initialized to zeros
- For sequence training, recurrent state has batch dimension only (not batch_length)

**For Each Timestep in Sequence**
1. Pass recurrent state to prior net → get prior
2. Pass recurrent state + encoded observation to posterior net → get posterior
3. Concatenate recurrent state + posterior → full state (uses posterior because more informed)
4. Pass full state to:
   - Reward predictor → predict reward
   - Continue predictor → predict episode continuation
   - Decoder → reconstruct observation
5. Pass full state + action (from buffer) to recurrent model → new recurrent state

**Critical Details**
- Action comes from buffer (real action that was taken)
- Uses posterior (not prior) because training benefits from best available information
- Decoder reconstruction loss applies pressure to keep representations informative
- Sequence processing done in batches (batch_size × batch_length)

**Reshaping Trick for Processing**
- Collapse batch_length into batch dimension for encoder
- Process all timesteps through encoder efficiently
- Reshape back to original dimensions
- Reason: Encoder expects 4D input (batch, height, width, channels)

#### Behavior Training Phase (BT) Diagram

**Transition from WMT**
- Take full states from WMT (well-informed due to posterior)
- Pass directly to BT as initial states for imagination

**For Each Step in Imagination**
1. Pass full state to actor → get action
2. Pass full state + action to recurrent model → new recurrent state
3. Pass recurrent state to prior net → new prior (NOT posterior - no real observation during imagination)
4. Concatenate recurrent state + prior → new full state (yellow in diagram = imagined)
5. Pass full state to:
   - Reward predictor (not trained, guides actor)
   - Continue predictor (not trained, guides actor)
   - Decoder (not used, just shown for completeness)
   - Critic net → get value estimate (not trained, guides actor)
   - Actor net (for policy gradient training)

**Critical Difference from WMT**
- Action comes from actor (not buffer)
- Uses prior (not posterior) - NO real observation in imagination
- Recurrent model takes live actions and imagined states
- Imagination horizon: typically 15 steps

**Key Insight on Dots Connecting**
- WMT produces well-informed full states
- These states feed directly into BT as starting points
- BT then imagines trajectories starting from these grounded states
- All imagination predictions (reward, continue, etc.) guide but don't train actor/critic

**Rollout Process**
- For each full state from WMT, create separate imagination rollout
- All full states from WMT used (could be subsampled in theory)
- Each rollout produces sequence of imagined transitions
- These sequences used to train actor and critic

---

## Lambda Returns & Loss Calculations

### Lambda Return Concept

**Goal**: Compute TD(λ) returns - combines immediate rewards with bootstrapped value estimates

**Mathematical Formula**:
- Lambda return incorporates:
  - Rewards at each step
  - Current critic value estimates
  - Continue probabilities (discount factors)
  - Bootstrap from last timestep's value

**Calculation Direction**: Backward from episode end
- Final value = critic(last_state) (bootstrap)
- Work backwards using recurrence relation:
  - λ_t = γ * λ_{t+1} * continue_t + reward_t + (1 - λ) * value_t

### Implementation Comparison

#### Original Dreamer Implementation Issues
```
- Complex indexing
- Reverse lists multiple times
- Variable names like "last" and "dis" unclear
- Bootstrap shifted constantly
- Hard to understand at first glance
```

#### ShER Library Approach
```
- Still reverses lists
- Concatenates instead of stacking
- Calculates intermediate values unnecessarily
- Unclear optimization rationale
```

#### SimpleDreamer Approach
```
- Still uses reverse/stack pattern
- Similar complexity to original
- Index adjustments needed
```

#### Presenter's Implementation Philosophy
```python
# Create buffer for lambda values
lambda_buffer = []

# Bootstrap starts as last critic value
bootstrap = values[-1]

# Iterate backwards through timesteps
for t in reversed(range(T-1)):
    lambda_t = gamma * lambda_{t+1} * continue_t + reward_t + (1 - lambda) * value_t
    lambda_buffer.append(lambda_t)

# Results are in correct order due to backwards iteration
```

**Advantages**:
- No list reversals needed
- Directly implements the mathematical equation
- Easy to follow - equation → code mapping
- Doesn't require passing device/horizon - inferred from data shapes
- More understandable and maintainable

---

## Implementation Comparison

### Why Multiple Implementations Exist

**Original Dreamer Code Challenges**:
- Supports multiple frameworks (JSON, TensorBoard, Weights&Biases, etc.)
- Support for many different environments and configurations
- Extreme generality makes it hard to read
- Reddit quote: Someone spent 3 months trying to understand it and was still overwhelmed

### Presenter's Philosophy: "Natural Dreamer"

**Naming Rationale**:
- Wanted to emphasize: "natural" like showing up naturally
- Initially considered: "Underdog Dreamer", "Amateur Dreamer"
- Not a professional/industry implementation
- Just someone who studied it and wrote clean code

**Analogy**:
- Other implementations use "super macro lenses for anti-polar, anti-reflective laser microscopy"
- Natural Dreamer: "Shoots to hit the middle things"
- Aims for clarity and learnability, not maximum performance

**Key Design Principle**: Remove everything not essential for core algorithm understanding
- Delete metrics initially
- Delete checkpointing code
- Delete unnecessary abstractions
- Keep only: collect data → train world model → train behavior

### Repository Quality
- Made nice for tutorial with code organization
- First commit contains exact code for tutorial
- Can trace git history to see changes/improvements
- Multiple attempts and failures documented in commit history

---

## Code Structure Overview

### Main Script Organization

#### Entry Point
```python
if __name__ == "__main__":
    # Parse arguments with config name
    # Execute main training loop
```

**Philosophy**: Simple entry - just pass config name, works out of the box

#### Configuration System
- YAML files with nested structure
- Convert to `Attrdict` - enables attribute access like `config.seed`
- Nested access like `config.dreamer.batch_size`
- Single config per run (easy to modify for experiments)

#### Reproducibility Setup
```python
# Set all random seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```
**Reason**: Eliminate randomness effects to see impact of specific changes

#### Folder Structure Setup
- Create folders: `metrics/`, `plots/`, `checkpoints/`, `videos/`
- Generate run names with environment + custom name
- Create filename bases for appending suffixes (2K, 4K, 8K steps)
- Automatically create directories to prevent errors

#### Environment Creation (Unique Approach)
```python
# Base environment - no rendering
env = gymnasium.make(env_name)

# Separate eval environment - with rendering
eval_env = gymnasium.make(env_name, render_mode="rgb_array")
```

**Rationale**:
- Training env as fast as possible (no unnecessary rendering)
- Eval env can render frames for video recording
- Belief: skipping render mode internal logic makes training faster
- Unusual but potentially justified optimization

#### Environment Properties Extraction
```python
channels, height, width = observation_space.shape
action_size = action_space.shape[0]
action_boundaries = (action_space.low, action_space.high)
```
**Note**: Action boundaries vary by environment (e.g., steering -1 to 1, gas 0 to 1)

#### Agent Instantiation
```python
config_dreamer = config.dreamer  # Pass subset of config
agent = Dreamer(
    env_props,
    config_dreamer
)
```

**Philosophy**: Each component gets only its relevant config section for clean isolation

#### Checkpoint Resume Logic
```python
if config.resume:
    agent.load(checkpoint_dir / f"{config.run_name}_{config.resume_suffix}")
```

### Training Loop Structure

#### Three-Level Nesting

**Level 1: Environment Steps Loop**
```python
while total_steps < max_steps:
    # Run episode
    # Accumulate steps
```

**Level 2: Training Iterations (Replay Ratio)**
```python
# For each environment step, perform multiple training steps
train_iterations = replay_ratio  # e.g., 100 (from original Dreamer)
for _ in range(train_iterations):
    # Sample batch from replay buffer
    # Train world model
    # Get initial states from world model
    # Train behavior using initial states
    # Increment gradient steps

    # Checkpoint logic (every N gradient steps)
    if gradient_steps % checkpoint_interval == 0:
        # Save checkpoint
        # Evaluate on eval env
        # Save video
```

**Level 3: Warmup Phase**
```python
# Before training, collect episodes without training
# Provides buffer diversity
# Typically ~few episodes
```

#### Replay Ratio Tuning
- Default: 100 (from paper)
- Can increase to improve sample efficiency
- Trades off training time for better learning from same data
- Formula: `train_iterations = (steps_since_last_train) * replay_ratio`

#### Checkpoint Management
```python
# Every N gradient steps
suffix = gradient_steps // 1000  # Convert to thousands (add "k" for short names)
checkpoint_path = checkpoint_dir / f"{base_name}_{suffix}k"
agent.save(checkpoint_path)
```

#### Metrics Collection
```python
# Every N training batches
base_info = {
    'env_steps': total_steps,
    'grad_steps': gradient_steps,
    'score': latest_eval_score
}

# Combine with:
world_model_metrics
actor_critic_metrics

# Save to CSV
csv_metrics.append({**base_info, **world_model_metrics, **actor_critic_metrics})
```

**Available Metrics**:
- `kl_loss`, `actor_loss`, `critic_loss`
- `advantages`
- And many others for debugging

#### Visualization
```python
# Create interactive HTML plot from CSV metrics
# Can toggle metrics on/off to inspect specific signals
# Check for:
  # - Training progress
  # - Convergence issues
  # - Overfitting (reward plateaus)
  # - Stability (smoothness of curves)
```

### Learning Strategy from Code

**Presenter's Approach to Learning**:
1. Start with someone else's code
2. Delete everything not essential for understanding
3. Understand core: collect data → train world model → train behavior
4. Branch out to specifics when needed
5. Read function definitions when hitting specifics
6. Don't force yourself to understand everything at once

---

## World Model Training

### Data Preparation & Reshaping

#### Batch Structure
```
Data shape: (batch_size=32, batch_length=64, channels=3, height=64, width=64)
Total: 32 sequences of 64 timesteps each
```

#### Reshaping Trick for Encoder
```python
# Encoder expects 4D input: (batch, height, width, channels)
# But we have: (batch_size, batch_length, channels, height, width)

# Solution: Collapse batch_length into batch dimension
reshaped = observations.reshape(batch_size * batch_length, channels, height, width)

# Pass through encoder
encoded = encoder(reshaped)

# Reshape back
encoded = encoded.reshape(batch_size, batch_length, embed_dim)
```

**Why**: Efficiently process all timesteps through single encoder pass

### Recurrent State & Latent State Initialization

```python
# Recurrent state: shape (batch_size, recurrent_dim)
# - No batch_length because it's one timestep per state in sequence
# - Initialized to zeros initially

# Latent state: same dimensionality but discrete/categorical
# - Also initialized to zeros

# Action: shape (batch_size,) - dummy for initialization step
```

**Key Insight**: These are for single timestep, not entire sequences yet

### Rollout Process During Training

#### Step-by-Step for Each Timestep
```
1. Pass recurrent_state to prior_net → prior_logits
   (not used in WMT, only needed for training)

2. Pass recurrent_state + encoded_observation to posterior_net → posterior_logits
   (used for training - better informed)

3. Sample from posterior distribution → posterior_sample

4. Concatenate recurrent_state + posterior_sample → full_state

5. Pass full_state to:
   - reward_predictor() → predict reward
   - continue_predictor() → predict episode continue
   - decoder() → reconstruct observation

6. Pass full_state + action_from_buffer to recurrent_model():
   - Process action through linear layer with activation
   - Combined with latent states and actions in processed form
   - Feed to GRU-like recurrent network
   - Output: new_recurrent_state for next timestep

7. Append all outputs to lists for later processing
```

#### After Full Sequence Processed
```python
# Stack all timesteps together
recurrent_states = stack(recurrent_states_list, dim=1)  # → (batch, T, rec_dim)
posteriors = stack(posteriors_list, dim=1)              # → (batch, T, lat_dim)
rewards = stack(rewards_list, dim=1)                    # → (batch, T)

# Concatenate to create full states
full_states = concatenate([recurrent_states, posteriors], dim=-1)
```

### Loss Calculations

#### 1. Reconstruction Loss
```python
# Decoder outputs means of normal distributions (std=1)
# Maximize probability of reconstructing real observations

reconstruction_loss = -log_probability(
    observations,
    means=decoder_output,
    std=1.0
)

# Mean over independent dimensions
reconstruction_loss = reconstruction_loss.mean()
```

**Significance**: Keeps representations informative by applying pressure through reconstruction

#### 2. Reward Loss
```python
# Reward predictor returns distribution
# Maximize probability of actual rewards

reward_loss = -reward_distribution.log_prob(rewards)

# Ignore first timestep (no initial recurrent state for it)
reward_loss = reward_loss[:, 1:].mean()
```

#### 3. KL Divergence Loss (Most Complex)

**Concept**: Make prior (prediction without observation) match posterior (with observation)

```python
# Create four categorical distributions
prior_dist = Categorical(logits=prior_logits)
posterior_dist = Categorical(logits=posterior_logits)

# Detached versions (no gradients through certain paths)
prior_dist_detached = Categorical(logits=prior_logits.detach())
posterior_dist_detached = Categorical(logits=posterior_logits.detach())

# Calculate KL divergence both ways
kl_dynamics = kl_divergence(prior_dist_detached, posterior_dist)
kl_representation = kl_divergence(posterior_dist_detached, prior_dist)

# Apply threshold: max(1, kl_value) - free bits
kl_dynamics = max(1.0, kl_dynamics)  # Free nats = 1.0
kl_representation = max(1.0, kl_representation)

# Apply weights and sum
# (Dynamics gets more weight to push prior toward posterior)
kl_loss = 1.0 * kl_dynamics + 0.1 * kl_representation
```

**Why Different Directions & Weights?**
- `kl_dynamics`: Dynamics predictor learning to match posterior
  - Gradient stops at posterior (posterior doesn't change from this)
  - Prior learns heavily
- `kl_representation`: Posterior learning to match prior
  - Gradient stops at prior (prior doesn't change from this)
  - Posterior learns slightly
- Asymmetric weighting (1.0 vs 0.1) biases learning toward prior

**Max with Free Nats**:
- If KL < threshold (1.0 nats), set to threshold
- Prevents degenerate solutions where posterior=prior trivially
- Ensures meaningful representations

**Importance**: This is the most complex loss and most important for world model quality

#### 4. Continue Loss (Optional)
```python
# Predict whether episode continues
continue_distribution = Bernoulli(probs)
continue_loss = -continue_distribution.log_prob(continues_real)
```

**Note**: Can optionally add to total world model loss

#### Total World Model Loss
```python
total_loss = reconstruction_loss + reward_loss + kl_loss + continue_loss (optional)
total_loss.backward()

# Step optimizer on all nets:
# - Encoder
# - Decoder
# - Prior net
# - Posterior net
# - Recurrent model
# - Reward predictor
# - Continue predictor
# - (NOT actor/critic - frozen)
```

### Output from World Model Training

```python
# Extract full states for behavior learning
initial_states = concatenate([rec_states, posteriors], dim=-1)

# Detach - no gradients flow back to world model during behavior training
initial_states = initial_states.detach()

# Reshape to remove batch_length dimension (flatten to single dimension)
initial_states = initial_states.reshape(-1, state_dim)

# Return both:
# 1. Metrics dict (for logging)
# 2. Initial states (for behavior training)
return metrics, initial_states
```

### Key Design Decisions

**Posterior-Based Training**: Always use posterior (more informed) during WMT
- Better quality targets for behavior learning
- Prior is guided toward posterior through KL loss

**Zero Initial State**: Recurrent state starts at zero
- No prior history
- Forces model to learn from observation

**Free Nats (Free Bits)**: KL loss threshold prevents trivial solutions
- Without it: posterior could always match prior → no learning
- With it (1.0 nats): Ensures minimum information flow

---

## Behavior Training

### Input to Behavior Training

```python
# Full states from world model training
# Shape: (N_initial_states, state_dim)
# These are well-informed because computed with posterior

# Split into components:
recurrent_states = full_states[:, :recurrent_dim]
latent_states = full_states[:, recurrent_dim:]
```

### Imagination Rollout Process

#### Initial Setup
```python
# For each initial full state, create separate imagination trajectory
# Initialize lists to collect predictions

for horizon_step in range(imagination_horizon):  # typically 15
    # 1. Get action from actor
    action = actor(full_state)

    # 2. Process through recurrent model
    # Note: uses prior (NO posterior - no real observation)
    new_recurrent = recurrent_model(
        recurrent_state,
        action,
        latent_state
    )

    # 3. Get prior latent state
    prior_latent = prior_net(new_recurrent)

    # 4. Combine into new full state
    new_full_state = concatenate([new_recurrent, prior_latent], dim=-1)

    # 5. Get predictions (for training actor/critic)
    reward = reward_predictor(new_full_state)
    continue_prob = continue_predictor(new_full_state)
    value = critic(new_full_state)

    # 6. Collect for later
    rewards_list.append(reward)
    values_list.append(value)
    continues_list.append(continue_prob)
    full_states_list.append(new_full_state)

    # 7. Update state for next step
    recurrent_state = new_recurrent
    latent_state = prior_latent
    full_state = new_full_state
```

#### Key Differences from World Model Training
- **Action Source**: Actor (live) vs Buffer (recorded)
- **Latent State**: Prior (no observation) vs Posterior (with observation)
- **What's Used**: Predictions guide learning, not trained
- **What's Trained**: Only actor and critic

#### Stacking for Batch Processing
```python
# After collecting for all horizon steps
rewards = stack(rewards_list, dim=1)      # (batch, horizon)
values = stack(values_list, dim=1)        # (batch, horizon)
continues = stack(continues_list, dim=1)  # (batch, horizon)
full_states = stack(full_states_list, dim=1)  # (batch, horizon, state_dim)

# Note: Index out first timestep (where we started)
rewards = rewards[:, 1:]      # (batch, horizon-1)
values = values[:, 1:]
continues = continues[:, 1:]
```

### Value Prediction Process

```python
# Compute lambda returns (go backward from end)
lambda_returns = compute_lambda_returns(
    rewards=rewards,           # (batch, horizon-1)
    values=values[:, :-1],     # (batch, horizon-1) - exclude last
    continues=continues,       # (batch, horizon-1)
    bootstrap=values[:, -1],   # Last value for bootstrapping
    gamma=0.997,
    lambda=0.95
)
```

#### Value Normalization
```python
# Estimate return scale using exponential moving average
scale = value_moments.estimate_scale(lambda_returns)

# Returns: (5th_percentile, 95th_percentile) using 99% decay rate
# Compute inverse scale for normalization

# Normalize both lambda returns and values
normalized_targets = (lambda_returns - mean) / (std + eps)
normalized_values = (values - mean) / (std + eps)

# Both advantages and values use same normalization
advantages = normalized_targets - normalized_values
```

### Actor Loss Computation

```python
# Maximize probability of advantageous actions
# Shape: (batch, horizon-1) - ignore last timestep

log_probs = actor(full_states).log_prob(actions)
policy_loss = -(log_probs * advantages.detach()).mean()

# Entropy bonus for exploration
entropy = actor(full_states).entropy().mean()
entropy_loss = -entropy_scale * entropy

actor_loss = policy_loss + entropy_loss
actor_loss.backward()
optimizer_actor.step()
```

**Why Detach on Advantage?**
- Advantages computed from critic (which has gradients)
- Actor shouldn't receive critic gradients
- Only actor policy gradients flow back

**Entropy Bonus**:
- Prevents policy from becoming too deterministic
- Coefficient controls exploration intensity

### Critic Loss Computation

```python
# Maximize probability of matching lambda returns
value_distribution = critic(full_states)  # Returns distribution
targets = lambda_returns.detach()

critic_loss = -value_distribution.log_prob(targets).mean()
critic_loss.backward()
optimizer_critic.step()
```

### Imagination Horizon

```python
# Default: 15 steps
# Affects:
# - Credit assignment window
# - Computation cost
# - Learning efficiency

# Can be tuned:
# - Lower: faster computation, shorter-term credit assignment
# - Higher: better credit assignment, more computation
```

---

## Neural Network Details

### Recurrent Model (GRU-like)

#### Structure
```python
# Takes:
# - recurrent_state: (batch, recurrent_dim)
# - latent_state: (batch, latent_dim)
# - action: (batch, action_dim)

# Process:
1. Concatenate latent_state + action → processed_input
2. Pass through linear layer → hidden
3. Apply activation (typically ELU)
4. Concatenate with recurrent_state → full_input
5. Pass to GRU cell → new_recurrent_state

# Output: new_recurrent_state
```

**Implementation Note**:
- Linear layer processes non-recurrent inputs
- GRU processes combined input
- Exception in the architecture (other inputs are separated)

### Prior Net & Posterior Net

#### Identical Structure, Different Inputs
```python
class LatentNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_categories, num_classes):
        self.mlp = build_mlp(
            input_size,
            num_categories * num_classes,  # output size
            [hidden_size] * num_layers
        )
        self.num_categories = num_categories
        self.num_classes = num_classes
```

#### Input Sizes
```python
# Prior Net
prior_input = recurrent_size  # 256 (example)

# Posterior Net
posterior_input = recurrent_size + encoded_obs_size  # 256 + 1024 = 1280
```

#### Forward Pass
```python
def forward(self, input_state):
    # Pass through MLP
    logits = self.mlp(input_state)  # → (batch, num_categories * num_classes)

    # Reshape to matrix
    logits = logits.reshape(batch, num_categories, num_classes)

    # Create categorical distributions (one for each category)
    # Apply uniform mixing trick (1% uniform, 99% network)
    uniform_dist = ones(num_classes) / num_classes

    # Mix probabilities
    probs = 0.99 * softmax(logits) + 0.01 * uniform_dist

    # Convert back to logits
    new_logits = log(probs)

    # Create distributions
    distributions = [Categorical(logits=new_logits[:, i])
                     for i in range(num_categories)]

    # Sample using reparameterization
    samples = [dist.rsample() for dist in distributions]

    # Flatten to vector
    output = concatenate(samples, dim=-1)  # → (batch, num_categories * num_classes)

    return output, logits  # Return both sample and logits
```

### Uniform Mixing Trick

**Purpose**: Prevent spikes in KL loss

**Paper Observation**:
- Without mixing: occasional KL loss spikes observed
- With 1% uniform mixing: smoother training

**Implementation**:
```python
uniform_prob = ones(num_classes) / num_classes  # Uniform distribution
network_prob = softmax(logits)  # Network output

mixed_prob = 0.99 * network_prob + 0.01 * uniform_prob

# Convert to logits for distribution
mixed_logits = log(mixed_prob)
```

**Questions Remaining**:
- Why does this help? (Not fully explained in paper)
- Does performance actually improve? (Worth testing)
- Could be ablated out for simplicity

### Model Network

```python
class ModelNet(nn.Module):
    # For reward and continue predictors

    def __init__(self, state_dim):
        self.net = build_mlp(state_dim, 2, [hidden_size] * num_layers)
        # Output 2 values: mean and log_std
```

#### Reward Predictor
```python
def forward(self, full_state):
    # Output: (mean, log_std) for normal distribution
    mean, log_std = self.net(full_state).chunk(2, dim=-1)
    std = exp(log_std)

    # Create normal distribution
    dist = Normal(mean, std)
    return dist
```

#### Critic Network (Same Architecture)
```python
# Same as reward predictor
# Returns normal distribution of expected returns
```

**Assumption**: Rewards and values follow normal distribution
- Works well in practice
- Sufficient for most environments
- Not universal like two-hot encoding

### Two-Hot Encoding (Not Implemented)

**Original Dreamer Implementation**:
- Uses bins to represent continuous values
- More complex than normal distribution
- Can model multi-modal distributions
- Bins: 255 categories from -20 to 20 (in log scale, then exponentiated)

**Status in Presenter's Code**:
- Implemented but doesn't work yet
- Everything coded correctly but performance not improving
- Worth investigating but not critical for MVP

**Tradeoff**:
- Normal dist: simpler, good enough, works
- Two-hot: more complex, theoretically better, not working yet

### Continue Predictor Network

```python
class ContinueNet(nn.Module):
    def __init__(self, state_dim):
        self.net = build_mlp(state_dim, 1, [hidden_size] * num_layers)
```

#### Forward
```python
def forward(self, full_state):
    # Single output: probability
    logit = self.net(full_state)

    # Create Bernoulli distribution (binary outcome)
    dist = Bernoulli(logits=logit)
    return dist
```

**Note**: Currently not used in training (continue_prob set to 1.0)

### Encoder Network

#### Purpose
- Convert observations to embeddings
- Universal "choke point" for observation processing
- Can standardize different observation types

#### Image Encoder (Current)
```python
class Encoder(nn.Module):
    def __init__(self, channels, embed_dim):
        # CNN for images
        self.layers = [
            Conv2d(in_channels, hidden, kernel_size=4, stride=2),
            Conv2d(hidden, hidden, kernel_size=4, stride=2),
            Conv2d(hidden, hidden, kernel_size=4, stride=2),
            Conv2d(hidden, hidden, kernel_size=4, stride=2),
        ]
        # Flatten and project to embed_dim
```

#### Numerical Observation Encoder (Future)
```python
# Simple MLP for vector observations
# Not yet implemented
```

#### Decoder

```python
class Decoder(nn.Module):
    # Takes full state
    # Outputs reconstructed observation
    # Assumed normal distribution with std=1
```

---

## Key Design Decisions

### Why Separate Posterior and Prior Networks?

**Alternative**: Single network that sometimes sees observation, sometimes doesn't
**Problem**: Would learn two different behaviors in same weights - confusing

**Solution**: Two separate networks
- Posterior: always has observation
- Prior: never has observation
- Connected through KL loss ensuring they learn similar features

### Why Discrete Representations?

**Advantage over Continuous**:
- Better empirical performance
- Prevents representation collapse
- Enables categorical distributions
- More stable training

**Tradeoff**: Less intuitive, requires categorical sampling

### Why Decoder Reconstruction?

**Not Used Directly For**:
- Policy training
- Value prediction
- Reward prediction

**Used For**:
- Regularization through loss signal
- Keeps representations informative
- Prevents information loss in compression

**Alternative Question**: What if removed?
- Worth ablating in future
- Paper doesn't explain necessity clearly
- But they use it, so we use it

### Why Teacher-Student (Encoder-Dynamics)?

**Problem**: During imagination, can't access real observations
- Dynamics predictor must predict next state from recurrent + latent only
- Directly training this predicts bad representations

**Solution**:
- Encoder (teacher) creates target representations with full information
- Dynamics (student) learns to match encoder through KL loss
- Both perspectives inform representation learning
- Student becomes useful for imagination

### Imagination Horizon Choice (15)

**Tradeoff**:
- Longer horizon: better credit assignment, higher computation
- Shorter horizon: faster training, myopic policy

**Paper Choice**: 15 steps
- Empirically found to balance these
- Could be hyperparameter to tune

### Lambda Return Computation

**Why Backwards**?**
- Bootstrap value from episode end
- Work backwards incorporating rewards and continues
- Gives clean recurrence relation

**Why λ parameter?**
- λ=0: Pure bootstrapping (critic values only, no rewards)
- λ=1: Monte Carlo (rewards only, no bootstrapping)
- λ=0.95: Balance both

### Entropy Regularization

**Why Needed?**
- Without it: actor converges to deterministic policy early
- Prevents exploration in imagination
- Creates boring, suboptimal policies

**How Much?**
- Default: 3e-3 (coefficient)
- Could be tuned per environment
- Higher: more exploration
- Lower: more exploitation

### Value Normalization (Return Moments)

**Purpose**: Stabilize advantage computation
- Raw advantages can have huge scale variation
- Normalization makes training more stable

**Method**: Exponential moving average of 5th-95th percentile
- Robust to outliers (unlike mean/std)
- Adapts over time
- Decay rate: 0.99

### Why Detach Advantages in Actor Loss?

```python
# Critic produces values with gradients
values = critic(states)  # gradients flow through critic weights

# Compute advantages
advantages = targets - values  # advantages have gradients through critic

# In actor loss:
log_probs = actor(states).log_prob(actions)
actor_loss = -(log_probs * advantages.detach()).mean()
```

**Reason**:
- Actor should only receive gradient from policy (action probability)
- Not from critic network
- Prevents feedback loop: critic → advantages → actor gradients affecting critic
- Each learns independently

---

## Summary of Learning Journey

### Recommended Learning Path

1. **Understand Three-Step Process**
   - Collect data
   - Train world model
   - Train behavior

2. **Study Architecture Diagrams**
   - Posterior/Prior split
   - Recurrent + Latent = Full State
   - What gets trained vs frozen

3. **Implement Simple Version**
   - Lambda returns
   - Actor/critic training
   - World model training

4. **Add Complexity Gradually**
   - Discrete representations
   - KL divergence weighting
   - Uniform mixing
   - Continue prediction

5. **Optimize and Tune**
   - Hyperparameters
   - Network architectures
   - Training procedures

### Key Takeaways

- **World Model**: Learns to predict dynamics, rewards, episode continuation
- **Behavior Model**: Learns policy entirely in imagination without real observations
- **Elegance**: Separated concerns - representation learning vs policy learning
- **Complexity**: Many details matter (KL weighting, free nats, value normalization)
- **Code Clarity**: Original implementation complex; simplified versions exist
- **Learnability**: Start simple, understand core, add details gradually

### Open Questions (Worth Investigating)

1. What happens without decoder reconstruction?
2. Does uniform mixing actually help performance?
3. What's optimal imagination horizon per environment?
4. Can two-hot encoding be made to work?
5. How much does continue prediction help?
6. Is value normalization necessary?
7. What's the impact of free nats threshold value?

---

## Conclusion

DreamerV3 represents sophisticated RL through world models. The key insight is separating **representation learning** (world model) from **behavior learning** (actor-critic). This allows agents to learn general understanding of dynamics, then efficiently train policies through imagination.

The architecture balances:
- **Complexity**: Many components must work together
- **Clarity**: Core concept (dream trajectories) is simple
- **Performance**: State-of-art results on 150+ tasks
- **Learnability**: Simplified implementations help understanding

Success depends on getting many details right:
- KL divergence balance between prior and posterior
- Value normalization for stable advantage estimates
- Entropy regularization for exploration
- Imagination horizon for credit assignment

The code walkthrough emphasizes: **Start simple, understand core, add details as you go.** This approach leads to clear implementations that are easier to debug, modify, and improve upon.
