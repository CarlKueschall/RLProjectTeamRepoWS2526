import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import yaml
import os
import attridict
import gymnasium as gym
import csv
import pandas as pd
import plotly.graph_objects as pgo


def seedEverything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def findFile(filename):
    currentDir = os.getcwd()
    for root, dirs, files in os.walk(currentDir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"File '{filename}' not found in subdirectories of {currentDir}")


def loadConfig(config_path):
    if not config_path.endswith(".yml"):
        config_path += ".yml"
    config_path = findFile(config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return attridict(config)


def getEnvProperties(env):
    observationShape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discreteActionBool = True
        actionSize = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discreteActionBool = False
        actionSize = env.action_space.shape[0]
    else:
        raise Exception
    return observationShape, discreteActionBool, actionSize


def saveLossesToCSV(filename, metrics):
    fileAlreadyExists = os.path.isfile(filename + ".csv")
    with open(filename + ".csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if not fileAlreadyExists:
            writer.writerow(metrics.keys())
        writer.writerow(metrics.values())


def plotMetrics(filename, title="", savePath="metricsPlot", window=10):
    if not filename.endswith(".csv"):
        filename += ".csv"
    
    data = pd.read_csv(filename)
    fig = pgo.Figure()

    colors = [
        "gold", "gray", "beige", "blueviolet", "cadetblue",
        "chartreuse", "coral", "cornflowerblue", "crimson", "darkorange",
        "deeppink", "dodgerblue", "forestgreen", "aquamarine", "lightseagreen",
        "lightskyblue", "mediumorchid", "mediumspringgreen", "orangered", "violet"]
    num_colors = len(colors)

    for idx, column in enumerate(data.columns):
        if column in ["envSteps", "gradientSteps"]:
            continue
        
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=data[column], mode='lines',
            name=f"{column} (original)",
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5, visible='legendonly'))
        
        smoothed_data = data[column].rolling(window=window, min_periods=1).mean()
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=smoothed_data, mode='lines',
            name=f"{column} (smoothed)",
            line=dict(color=colors[idx % num_colors], width=2)))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=30),
            yanchor='top'
        ),
        xaxis=dict(
            title="Gradient Steps",
            showgrid=True,
            zeroline=False,
            position=0
        ),
        yaxis_title="Value",
        template="plotly_dark",
        height=1080,
        width=1920,
        margin=dict(t=60, l=40, r=40, b=40),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="White",
            borderwidth=2,
            font=dict(size=12)
        )
    )

    if not savePath.endswith(".html"):
        savePath += ".html"
    fig.write_html(savePath)


def sequentialModel1D(inputSize, hiddenSizes, outputSize, activationFunction="Tanh", finishWithActivation=False):
    activationFunction = getattr(nn, activationFunction)()
    layers = []
    currentInputSize = inputSize

    for hiddenSize in hiddenSizes:
        layers.append(nn.Linear(currentInputSize, hiddenSize))
        layers.append(activationFunction)
        currentInputSize = hiddenSize
    
    layers.append(nn.Linear(currentInputSize, outputSize))
    if finishWithActivation:
        layers.append(activationFunction)

    return nn.Sequential(*layers)


def computeLambdaValues(rewards, values, continues, lambda_=0.95):
    """
    Compute TD(λ) returns for imagination trajectories.

    Args:
        rewards: (B, H-1) predicted rewards
        values: (B, H) predicted values for states 0 to H-1
        continues: (B, H) continue probabilities (discount factors)
        lambda_: TD(λ) parameter

    Returns:
        (B, H-1) lambda returns for states 0 to H-2
    """
    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]
    for i in reversed(range(rewards.shape[-1])):
        # TD(λ): G_t = r_t + γ * ((1-λ) * V_{t+1} + λ * G_{t+1})
        # values[:, i+1] is V_{t+1} (next state's value)
        returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i+1] + lambda_ * bootstrap)
        bootstrap = returns[:, i]
    return returns


def ensureParentFolders(*paths):
    for path in paths:
        parentFolder = os.path.dirname(path)
        if parentFolder and not os.path.exists(parentFolder):
            os.makedirs(parentFolder, exist_ok=True)


# =============================================================================
# Symlog Transform (DreamerV3)
# =============================================================================
# Symmetric logarithmic transform that compresses large values while
# preserving sign and behavior near zero. Essential for handling rewards
# that can range from -10 to +10 (or larger).
#
# symlog(x) = sign(x) * ln(|x| + 1)
# symexp(x) = sign(x) * (exp(|x|) - 1)
#
# Properties:
#   - symlog(0) = 0
#   - symlog(±10) ≈ ±2.4
#   - symlog(±1000) ≈ ±6.9
#   - Gradient near 0 is ~1 (well-behaved)
# =============================================================================

def symlog(x):
    """Symmetric logarithmic transform: compresses large values."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    """Inverse of symlog: expands compressed values back."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# =============================================================================
# Two-Hot Symlog Encoding (DreamerV3)
# =============================================================================
# Instead of predicting continuous values with a Normal distribution,
# DreamerV3 discretizes values into bins and predicts a categorical
# distribution. The "two-hot" encoding spreads probability mass between
# the two adjacent bins that bracket the target value.
#
# This is crucial for:
#   1. Handling multi-modal distributions (rewards are often 0 OR ±10)
#   2. Avoiding regression-to-mean issues with Normal distributions
#   3. Better gradient signal for rare events (sparse rewards)
#
# Process:
#   1. symlog(value) -> compressed value in ~[-20, 20]
#   2. Find two adjacent bins that bracket the value
#   3. Split probability between them based on distance
#   4. Train with cross-entropy loss
#   5. Decode: weighted sum of bins -> symexp -> original scale
# =============================================================================

class TwoHotSymlog(nn.Module):
    """
    Two-Hot Symlog encoding for reward and value prediction.

    Args:
        bins: Number of discrete bins (default: 255)
        min_val: Minimum value in symlog space (default: -20)
        max_val: Maximum value in symlog space (default: +20)

    Usage:
        twohot = TwoHotSymlog(bins=255)

        # Training: compute loss
        loss = twohot.loss(logits, target_values)

        # Inference: decode to scalar
        values = twohot.decode(logits)
    """

    def __init__(self, bins=255, min_val=-20.0, max_val=20.0):
        super().__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val

        # Create evenly-spaced bin centers in symlog space
        # bins[0] = -20, bins[127] ≈ 0, bins[254] = +20
        self.register_buffer("bin_centers", torch.linspace(min_val, max_val, bins))

        # Step size between bins
        self.step = (max_val - min_val) / (bins - 1)

    def loss(self, logits, target):
        """
        Compute two-hot cross-entropy loss.

        Args:
            logits: Network output, shape (*, bins)
            target: Target scalar values, shape (*,) or (*, 1)

        Returns:
            Loss tensor, shape (*,)
        """
        # Flatten target if needed
        if target.dim() > logits.dim() - 1:
            target = target.squeeze(-1)

        # Transform target to symlog space
        y = symlog(target)

        # Clamp to bin range
        y = torch.clamp(y, self.min_val, self.max_val)

        # Calculate continuous bin index (0 to bins-1)
        # e.g., y=-20 -> 0, y=0 -> 127, y=+20 -> 254
        continuous_idx = (y - self.min_val) / self.step

        # Lower and upper bin indices
        k = continuous_idx.long()
        k = torch.clamp(k, 0, self.bins - 2)  # Ensure k+1 is valid
        k_plus_1 = k + 1

        # Weight for upper bin (how far we are from lower bin)
        alpha = continuous_idx - k.float()
        alpha = torch.clamp(alpha, 0.0, 1.0)

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for the two adjacent bins
        log_p_k = log_probs.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        log_p_k1 = log_probs.gather(-1, k_plus_1.unsqueeze(-1)).squeeze(-1)

        # Two-hot cross-entropy: weighted sum of log probs
        # target_probs[k] = 1 - alpha, target_probs[k+1] = alpha
        loss = -((1 - alpha) * log_p_k + alpha * log_p_k1)

        return loss

    def decode(self, logits):
        """
        Decode logits to scalar values.

        Args:
            logits: Network output, shape (*, bins)

        Returns:
            Scalar values, shape (*,)
        """
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        # Weighted sum of bin centers (expected value in symlog space)
        y_hat = torch.sum(probs * self.bin_centers, dim=-1)

        # Transform back to original scale
        return symexp(y_hat)

    def encode_target(self, target):
        """
        Encode target values as two-hot probability distributions.
        Useful for visualization/debugging.

        Args:
            target: Target scalar values, shape (*,)

        Returns:
            Two-hot probabilities, shape (*, bins)
        """
        y = symlog(target)
        y = torch.clamp(y, self.min_val, self.max_val)

        continuous_idx = (y - self.min_val) / self.step
        k = continuous_idx.long()
        k = torch.clamp(k, 0, self.bins - 2)

        alpha = continuous_idx - k.float()
        alpha = torch.clamp(alpha, 0.0, 1.0)

        # Create two-hot distribution
        probs = torch.zeros(*target.shape, self.bins, device=target.device)
        probs.scatter_(-1, k.unsqueeze(-1), (1 - alpha).unsqueeze(-1))
        probs.scatter_add_(-1, (k + 1).unsqueeze(-1), alpha.unsqueeze(-1))

        return probs


class Moments(nn.Module):
    def __init__( self, device, decay = 0.99, min_=0.01, percentileLow = 0.05, percentileHigh = 0.95):
        super().__init__()
        self._decay = decay
        self._min = torch.tensor(min_)
        self._percentileLow = percentileLow
        self._percentileHigh = percentileHigh
        self.register_buffer("low", torch.zeros((), dtype=torch.float32, device=device))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()
        low = torch.quantile(x, self._percentileLow)
        high = torch.quantile(x, self._percentileHigh)
        self.low = self._decay*self.low + (1 - self._decay)*low
        self.high = self._decay*self.high + (1 - self._decay)*high
        inverseScale = torch.max(self._min, self.high - self.low)
        return self.low.detach(), inverseScale.detach()
