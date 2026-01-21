"""
DreamerV3 Networks for Hockey (Low-dimensional observations).

Based on NaturalDreamer, adapted for 18-dim vector observations.
Uses MLP encoder/decoder instead of CNN.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits
from utils import sequentialModel1D


class RecurrentModel(nn.Module):
    """GRU-based recurrent model for RSSM dynamics."""

    def __init__(self, recurrentSize, latentSize, actionSize, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)()
        self.linear = nn.Linear(latentSize + actionSize, self.config.hiddenSize)
        self.recurrent = nn.GRUCell(self.config.hiddenSize, recurrentSize)

    def forward(self, recurrentState, latentState, action):
        x = torch.cat((latentState, action), -1)
        x = self.activation(self.linear(x))
        return self.recurrent(x, recurrentState)


class PriorNet(nn.Module):
    """Prior network: predicts latent state from recurrent state only."""

    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength * latentClasses
        self.network = sequentialModel1D(
            inputSize,
            [self.config.hiddenSize] * self.config.numLayers,
            self.latentSize,
            self.config.activation
        )

    def forward(self, x):
        rawLogits = self.network(x)
        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities) / self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix) * probabilities + self.config.uniformMix * uniform
        logits = probs_to_logits(finalProbabilities)
        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits


class PosteriorNet(nn.Module):
    """Posterior network: infers latent state from recurrent state + observation."""

    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength * latentClasses
        self.network = sequentialModel1D(
            inputSize,
            [self.config.hiddenSize] * self.config.numLayers,
            self.latentSize,
            self.config.activation
        )

    def forward(self, x):
        rawLogits = self.network(x)
        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities) / self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix) * probabilities + self.config.uniformMix * uniform
        logits = probs_to_logits(finalProbabilities)
        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits


class RewardModel(nn.Module):
    """
    Reward predictor using Two-Hot Symlog encoding.

    Instead of predicting a Normal distribution (which fails for multi-modal
    rewards like 0 vs ±10), we predict a categorical distribution over 255 bins
    in symlog space. This allows the model to correctly represent sparse rewards.
    """

    def __init__(self, inputSize, config, bins=255):
        super().__init__()
        self.config = config
        self.bins = bins
        self.network = sequentialModel1D(
            inputSize,
            [self.config.hiddenSize] * self.config.numLayers,
            bins,  # Output logits for each bin
            self.config.activation
        )
        # Initialize output layer to zeros for uniform initial predictions
        # This makes initial reward predictions ≈ 0 (center bin in symlog space)
        # Critical for sparse rewards: prevents hallucinated rewards early in training
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.zero_()

    def forward(self, x):
        """Returns logits of shape (*, bins)."""
        return self.network(x)


class ContinueModel(nn.Module):
    """Continue predictor: outputs Bernoulli for episode continuation."""

    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(
            inputSize,
            [self.config.hiddenSize] * self.config.numLayers,
            1,
            self.config.activation
        )

    def forward(self, x):
        return Bernoulli(logits=self.network(x).squeeze(-1))


class EncoderMLP(nn.Module):
    """MLP Encoder for low-dimensional observations (e.g., 18-dim hockey state)."""

    def __init__(self, inputSize, outputSize, config):
        super().__init__()
        self.config = config
        self.outputSize = outputSize
        self.network = sequentialModel1D(
            inputSize,
            [self.config.hiddenSize] * self.config.numLayers,
            outputSize,
            self.config.activation,
            finishWithActivation=True  # End with activation for embedding
        )

    def forward(self, x):
        # x shape: (batch, obs_dim) or (batch * seq, obs_dim)
        return self.network(x).view(-1, self.outputSize)


class DecoderMLP(nn.Module):
    """MLP Decoder for low-dimensional observations."""

    def __init__(self, inputSize, outputSize, config):
        super().__init__()
        self.config = config
        self.outputSize = outputSize
        self.network = sequentialModel1D(
            inputSize,
            [self.config.hiddenSize] * self.config.numLayers,
            outputSize,
            self.config.activation,
            finishWithActivation=False  # Output is reconstruction mean
        )

    def forward(self, x):
        return self.network(x)


class Actor(nn.Module):
    """Actor network: outputs actions with tanh squashing."""

    def __init__(self, inputSize, actionSize, actionLow, actionHigh, device, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(
            inputSize,
            [self.config.hiddenSize] * self.config.numLayers,
            actionSize * 2,  # mean and log_std for each action dim
            self.config.activation
        )
        self.register_buffer("actionScale", (torch.tensor(actionHigh, device=device) - torch.tensor(actionLow, device=device)) / 2.0)
        self.register_buffer("actionBias", (torch.tensor(actionHigh, device=device) + torch.tensor(actionLow, device=device)) / 2.0)

    def forward(self, x, training=False):
        # logStdMin=-2 gives std_min=0.135, preventing entropy collapse
        # Old value of -5 allowed std=0.0067 which caused entropy to go negative
        logStdMin, logStdMax = -2, 2
        mean, logStd = self.network(x).chunk(2, dim=-1)
        # Bound log_std to [logStdMin, logStdMax]
        logStd = logStdMin + (logStdMax - logStdMin) / 2 * (torch.tanh(logStd) + 1)
        std = torch.exp(logStd)

        distribution = Normal(mean, std)
        sample = distribution.sample()
        sampleTanh = torch.tanh(sample)
        action = sampleTanh * self.actionScale + self.actionBias

        if training:
            logprobs = distribution.log_prob(sample)
            # Jacobian correction for tanh
            logprobs -= torch.log(self.actionScale * (1 - sampleTanh.pow(2)) + 1e-6)
            entropy = distribution.entropy()
            return action, logprobs.sum(-1), entropy.sum(-1)
        else:
            return action


class Critic(nn.Module):
    """
    Critic network using Two-Hot Symlog encoding.

    Instead of predicting a Normal distribution for values, we predict a
    categorical distribution over 255 bins in symlog space. This is critical
    for correctly predicting sharp value differences (e.g., "about to score"
    vs "just missed").

    Note: The output layer is initialized to zeros so initial predictions
    are uniform (value ≈ 0), following DreamerV3 paper.
    """

    def __init__(self, inputSize, config, bins=255):
        super().__init__()
        self.config = config
        self.bins = bins
        self.network = sequentialModel1D(
            inputSize,
            [self.config.hiddenSize] * self.config.numLayers,
            bins,  # Output logits for each bin
            self.config.activation
        )
        # Initialize output layer to zeros for uniform initial predictions
        # This makes initial value predictions ≈ 0
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.zero_()

    def forward(self, x):
        """Returns logits of shape (*, bins)."""
        return self.network(x)


# =============================================================================
# Auxiliary Task Heads
# =============================================================================
# These predict derived quantities that help the world model learn
# representations useful for goal prediction, without corrupting the reward signal.
# =============================================================================

class GoalPredictionHead(nn.Module):
    """
    Predicts probability of a goal occurring in the next K steps.

    This is the most directly relevant auxiliary task - it forces the latent
    state to encode "goal-likelihood" features without modifying rewards.

    Output: sigmoid probability (0 = no goal likely, 1 = goal imminent)
    """

    def __init__(self, inputSize, hiddenSize=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize // 2),
            nn.ReLU(),
            nn.Linear(hiddenSize // 2, 1)
        )

    def forward(self, x):
        """Returns logits for binary classification (use BCE with logits)."""
        return self.network(x).squeeze(-1)


class DistanceHead(nn.Module):
    """
    Predicts a scalar distance value (e.g., puck-to-goal, agent-to-puck).

    These spatial relationships are critical for understanding scoring dynamics.

    Output: predicted distance (regression, positive value)
    """

    def __init__(self, inputSize, hiddenSize=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, 1)
        )

    def forward(self, x):
        """Returns predicted distance (scalar)."""
        return self.network(x).squeeze(-1)


class ShotQualityHead(nn.Module):
    """
    Predicts a "shot quality" score combining position and velocity.

    High quality = puck moving toward goal from close range with clear path.
    This helps the model understand what makes a good scoring opportunity.

    Output: quality score (higher = better scoring chance)
    """

    def __init__(self, inputSize, hiddenSize=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize // 2),
            nn.ReLU(),
            nn.Linear(hiddenSize // 2, 1)
        )

    def forward(self, x):
        """Returns shot quality score (scalar)."""
        return self.network(x).squeeze(-1)