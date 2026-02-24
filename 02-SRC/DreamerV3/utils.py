"""
This file was developed with assistance from AI: autocomplete and discussion
about the contents and behavior of the code.
"""

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
    """TD(λ) returns for imagination. G_t = r_t + γ*((1-λ)*V_{t+1} + λ*G_{t+1})."""
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


# --- Symlog (DreamerV3) ---
# symlog(x) = sign(x)*ln(|x|+1), symexp is inverse. Compresses -10..+10 rewards.
# symlog(0)=0, symlog(±10)≈±2.4, gradient near 0 is ~1.

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# --- Two-Hot Symlog (DreamerV3) ---
# Discretize into bins instead of Normal. Two-hot spreads mass between adjacent bins.
# Handles multi-modal rewards (0 or ±10), avoids regression-to-mean, better gradients for sparse.

class TwoHotSymlog(nn.Module):

    def __init__(self, bins=255, min_val=-20.0, max_val=20.0):
        super().__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val

        self.register_buffer("bin_centers", torch.linspace(min_val, max_val, bins))

        self.step = (max_val - min_val) / (bins - 1)

    def loss(self, logits, target):
        if target.dim() > logits.dim() - 1:
            target = target.squeeze(-1)
        y = symlog(target)
        y = torch.clamp(y, self.min_val, self.max_val)
        continuous_idx = (y - self.min_val) / self.step
        k = continuous_idx.long()
        k = torch.clamp(k, 0, self.bins - 2)  # Ensure k+1 is valid
        k_plus_1 = k + 1
        alpha = continuous_idx - k.float()
        alpha = torch.clamp(alpha, 0.0, 1.0)
        log_probs = F.log_softmax(logits, dim=-1)
        log_p_k = log_probs.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        log_p_k1 = log_probs.gather(-1, k_plus_1.unsqueeze(-1)).squeeze(-1)
        loss = -((1 - alpha) * log_p_k + alpha * log_p_k1)

        return loss

    def decode(self, logits):
        probs = F.softmax(logits, dim=-1)
        y_hat = torch.sum(probs * self.bin_centers, dim=-1)
        return symexp(y_hat)

    def encode_target(self, target):
        y = symlog(target)
        y = torch.clamp(y, self.min_val, self.max_val)

        continuous_idx = (y - self.min_val) / self.step
        k = continuous_idx.long()
        k = torch.clamp(k, 0, self.bins - 2)

        alpha = continuous_idx - k.float()
        alpha = torch.clamp(alpha, 0.0, 1.0)
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
