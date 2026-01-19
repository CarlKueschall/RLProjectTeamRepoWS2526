"""
DreamerV3 Agent for Hockey (Low-dimensional observations).

Based on NaturalDreamer, adapted for 18-dim vector observations.
Simple, clean implementation following KISS principle.

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Independent, OneHotCategoricalStraightThrough, Normal
import numpy as np
import os

from networks import RecurrentModel, PriorNet, PosteriorNet, RewardModel, ContinueModel, EncoderMLP, DecoderMLP, Actor, Critic
from utils import computeLambdaValues, Moments, TwoHotSymlog, symlog
from buffer import ReplayBuffer


class Dreamer:
    """
    DreamerV3 agent for hockey environment.

    Key components:
    - World Model: encoder, decoder, recurrent model, prior/posterior nets, reward/continue predictors
    - Behavior: actor and critic trained in imagination

    Training flow:
    1. Collect experience in real environment
    2. Train world model on sequences from buffer
    3. Train actor-critic in imagined trajectories
    """

    def __init__(self, observationSize, actionSize, actionLow, actionHigh, device, config):
        """
        Args:
            observationSize: Dimension of observation vector (e.g., 18 for hockey)
            actionSize: Dimension of action vector (e.g., 4 for hockey)
            actionLow: Lower bounds for actions
            actionHigh: Upper bounds for actions
            device: torch device
            config: Configuration object with hyperparameters
        """
        self.observationSize = observationSize
        self.actionSize = actionSize
        self.config = config
        self.device = device

        # State dimensions
        self.recurrentSize = config.recurrentSize
        self.latentSize = config.latentLength * config.latentClasses
        self.fullStateSize = config.recurrentSize + self.latentSize

        # World model components
        self.encoder = EncoderMLP(observationSize, config.encodedObsSize, config.encoder).to(device)
        self.decoder = DecoderMLP(self.fullStateSize, observationSize, config.decoder).to(device)
        self.recurrentModel = RecurrentModel(config.recurrentSize, self.latentSize, actionSize, config.recurrentModel).to(device)
        self.priorNet = PriorNet(config.recurrentSize, config.latentLength, config.latentClasses, config.priorNet).to(device)
        self.posteriorNet = PosteriorNet(config.recurrentSize + config.encodedObsSize, config.latentLength, config.latentClasses, config.posteriorNet).to(device)
        self.rewardPredictor = RewardModel(self.fullStateSize, config.reward).to(device)

        if config.useContinuationPrediction:
            self.continuePredictor = ContinueModel(self.fullStateSize, config.continuation).to(device)

        # Behavior model
        self.actor = Actor(self.fullStateSize, actionSize, actionLow, actionHigh, device, config.actor).to(device)
        self.critic = Critic(self.fullStateSize, config.critic).to(device)

        # Replay buffer
        self.buffer = ReplayBuffer(observationSize, actionSize, config.buffer, device)

        # Two-Hot Symlog encoding for rewards and values
        # This is critical for handling sparse rewards (0 vs ±10)
        bins = getattr(config, 'twoHotBins', 255)
        min_val = getattr(config, 'twoHotMinVal', -20.0)
        max_val = getattr(config, 'twoHotMaxVal', 20.0)
        self.twoHot = TwoHotSymlog(bins=bins, min_val=min_val, max_val=max_val).to(device)
        self.twoHotBins = bins  # Store for use in behavior training

        # Value normalization
        self.valueMoments = Moments(device)

        # Collect world model parameters
        self.worldModelParameters = (
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.recurrentModel.parameters()) +
            list(self.priorNet.parameters()) +
            list(self.posteriorNet.parameters()) +
            list(self.rewardPredictor.parameters())
        )
        if config.useContinuationPrediction:
            self.worldModelParameters += list(self.continuePredictor.parameters())

        # Optimizers
        self.worldModelOptimizer = torch.optim.Adam(self.worldModelParameters, lr=config.worldModelLR)
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actorLR)
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=config.criticLR)

        # Counters
        self.totalEpisodes = 0
        self.totalEnvSteps = 0
        self.totalGradientSteps = 0

    def worldModelTraining(self, data):
        """
        Train world model on a batch of sequences.

        Args:
            data: Batch from replay buffer with observations, actions, rewards, dones

        Returns:
            fullStates: Initial states for behavior training (detached)
            metrics: Dictionary of training metrics
        """
        batchSize = self.config.batchSize
        batchLength = self.config.batchLength

        # Encode all observations (apply symlog first to handle velocity spikes)
        # data.observations: (batchSize, batchLength, obsSize)
        obsSymlog = symlog(data.observations)  # Compress large values
        encodedObs = self.encoder(obsSymlog.view(-1, self.observationSize))
        encodedObs = encodedObs.view(batchSize, batchLength, -1)

        # Initialize states
        h = torch.zeros(batchSize, self.recurrentSize, device=self.device)
        z = torch.zeros(batchSize, self.latentSize, device=self.device)

        # Collect states through sequence
        recurrentStates, priorsLogits, posteriors, posteriorsLogits = [], [], [], []

        for t in range(1, batchLength):
            # Step recurrent model with previous action
            h = self.recurrentModel(h, z, data.actions[:, t - 1])

            # Get prior (from h only) and posterior (from h + observation)
            _, priorLogits = self.priorNet(h)
            z, posteriorLogits = self.posteriorNet(torch.cat((h, encodedObs[:, t]), -1))

            recurrentStates.append(h)
            priorsLogits.append(priorLogits)
            posteriors.append(z)
            posteriorsLogits.append(posteriorLogits)

        # Stack collected states
        recurrentStates = torch.stack(recurrentStates, dim=1)      # (B, T-1, recurrentSize)
        priorsLogits = torch.stack(priorsLogits, dim=1)            # (B, T-1, latentLength, latentClasses)
        posteriors = torch.stack(posteriors, dim=1)                # (B, T-1, latentSize)
        posteriorsLogits = torch.stack(posteriorsLogits, dim=1)    # (B, T-1, latentLength, latentClasses)
        fullStates = torch.cat((recurrentStates, posteriors), dim=-1)  # (B, T-1, fullStateSize)

        # === Reconstruction Loss ===
        decodedObs = self.decoder(fullStates.view(-1, self.fullStateSize))
        decodedObs = decodedObs.view(batchSize, batchLength - 1, self.observationSize)
        # Target is symlog-transformed observations (decoder predicts in symlog space)
        obsTargets = obsSymlog[:, 1:]  # Use already-computed symlog observations
        reconDist = Independent(Normal(decodedObs, torch.ones_like(decodedObs)), 1)
        reconstructionLoss = -reconDist.log_prob(obsTargets).mean()

        # === Reward Prediction Loss (Two-Hot Symlog with Sparse Event Weighting) ===
        # Get logits from reward predictor (255 bins)
        rewardLogits = self.rewardPredictor(fullStates.view(-1, self.fullStateSize))
        rewardLogits = rewardLogits.view(batchSize, batchLength - 1, -1)  # (B, T-1, 255)
        rewardTargets = data.rewards[:, 1:].squeeze(-1)  # (B, T-1)

        # Compute per-sample two-hot loss
        rewardLossPerSample = self.twoHot.loss(rewardLogits, rewardTargets)  # (B, T-1)

        # === Inverse Frequency Weighting for Sparse Events ===
        # Sparse events (|reward| > 1) are ~1% of samples but carry critical signal.
        # Without weighting, they contribute ~1% of gradient and get drowned out.
        # With inverse frequency weighting, sparse and non-sparse contribute equally.
        SPARSE_THRESHOLD = 1.0
        sparseMask = rewardTargets.abs() > SPARSE_THRESHOLD
        numSparse = sparseMask.sum().float()
        numNonsparse = (~sparseMask).sum().float()

        # Compute weights: sparse events get weight = (num_nonsparse / num_sparse)
        # This makes total gradient contribution equal for both classes
        rewardWeights = torch.ones_like(rewardLossPerSample)
        sparseWeight = torch.tensor(1.0, device=rewardTargets.device)  # Default weight
        if numSparse > 0:
            # Weight sparse events by inverse frequency, capped at 100x to prevent instability
            sparseWeight = torch.clamp(numNonsparse / numSparse, min=1.0, max=100.0)
            rewardWeights[sparseMask] = sparseWeight

        # Weighted mean loss
        rewardLoss = (rewardLossPerSample * rewardWeights).sum() / rewardWeights.sum()

        # === KL Loss ===
        priorDist = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits), 1)
        priorDistSG = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits.detach()), 1)
        posteriorDist = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits), 1)
        posteriorDistSG = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits.detach()), 1)

        # Prior loss: train prior to match posterior (stop grad on posterior)
        priorLoss = kl_divergence(posteriorDistSG, priorDist)
        # Posterior loss: train posterior to match prior (stop grad on prior)
        posteriorLoss = kl_divergence(posteriorDist, priorDistSG)

        # Apply free nats threshold
        freeNats = torch.full_like(priorLoss, self.config.freeNats)
        priorLoss = self.config.betaPrior * torch.maximum(priorLoss, freeNats)
        posteriorLoss = self.config.betaPosterior * torch.maximum(posteriorLoss, freeNats)
        klLoss = (priorLoss + posteriorLoss).mean()

        # === Total World Model Loss ===
        worldModelLoss = reconstructionLoss + rewardLoss + klLoss

        # Continue prediction (optional)
        if self.config.useContinuationPrediction:
            continueDist = self.continuePredictor(fullStates)
            continueLoss = -continueDist.log_prob(1 - data.dones[:, 1:].squeeze(-1)).mean()
            worldModelLoss += continueLoss

        # Backprop
        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        worldModelGradNorm = nn.utils.clip_grad_norm_(self.worldModelParameters, self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.worldModelOptimizer.step()

        # === Detailed Metrics ===
        klLossShift = (self.config.betaPrior + self.config.betaPosterior) * self.config.freeNats

        # Reconstruction error per sample (for distribution stats)
        reconError = (decodedObs - obsTargets).pow(2).mean(dim=-1)  # (B, T-1)

        # Reward prediction accuracy (decode from two-hot logits)
        with torch.no_grad():
            predictedRewardMean = self.twoHot.decode(rewardLogits)  # (B, T-1)
        actualRewards = rewardTargets
        rewardPredError = (predictedRewardMean - actualRewards).abs()

        # Latent utilization - entropy of posterior categorical
        posteriorProbs = torch.softmax(posteriorsLogits, dim=-1)  # (B, T-1, latentLength, latentClasses)
        latentEntropy = -(posteriorProbs * (posteriorProbs + 1e-8).log()).sum(dim=-1).mean()  # avg entropy per variable

        # Prior-posterior agreement
        priorProbs = torch.softmax(priorsLogits, dim=-1)
        priorPosteriorKL = (posteriorProbs * ((posteriorProbs + 1e-8).log() - (priorProbs + 1e-8).log())).sum(dim=-1).mean()

        # === SPARSE REWARD SIGNAL METRICS ===
        # These tell us if the world model is learning from sparse rewards

        # Identify sparse reward events (non-zero rewards, typically ±10 for win/loss goals)
        SPARSE_THRESHOLD = 1.0  # Rewards above this magnitude are "sparse events"
        sparse_mask = actualRewards.abs() > SPARSE_THRESHOLD
        num_sparse_events = sparse_mask.sum().item()
        total_samples = actualRewards.numel()
        sparse_event_rate = num_sparse_events / total_samples

        # Reward prediction accuracy specifically on sparse events
        if num_sparse_events > 0:
            sparse_rewards = actualRewards[sparse_mask]
            sparse_predictions = predictedRewardMean[sparse_mask]
            sparse_pred_error = (sparse_predictions - sparse_rewards).abs().mean().item()
            sparse_pred_mean = sparse_predictions.mean().item()
            sparse_actual_mean = sparse_rewards.mean().item()
            # Does the model predict the RIGHT SIGN for sparse rewards?
            sparse_sign_accuracy = ((sparse_predictions * sparse_rewards) > 0).float().mean().item()
        else:
            sparse_pred_error = 0.0
            sparse_pred_mean = 0.0
            sparse_actual_mean = 0.0
            sparse_sign_accuracy = 0.0

        # Non-sparse (dense/PBRS) reward prediction
        nonsparse_mask = ~sparse_mask
        if nonsparse_mask.sum() > 0:
            nonsparse_pred_error = rewardPredError[nonsparse_mask].mean().item()
        else:
            nonsparse_pred_error = 0.0

        # Reward variance in buffer - if too low, no signal
        reward_variance = actualRewards.var().item()

        # Reward range in batch
        reward_min = actualRewards.min().item()
        reward_max = actualRewards.max().item()

        metrics = {
            # World model losses
            "world/loss": worldModelLoss.item() - klLossShift,
            "world/recon_loss": reconstructionLoss.item(),
            "world/reward_loss": rewardLoss.item(),
            "world/kl_loss": klLoss.item() - klLossShift,

            # Reconstruction quality
            "world/recon_error_mean": reconError.mean().item(),
            "world/recon_error_std": reconError.std().item(),
            "world/recon_error_max": reconError.max().item(),

            # Reward prediction quality (overall)
            "world/reward_pred_error_mean": rewardPredError.mean().item(),
            "world/reward_pred_error_std": rewardPredError.std().item(),
            "world/reward_pred_mean": predictedRewardMean.mean().item(),
            "world/reward_actual_mean": actualRewards.mean().item(),

            # Latent space health
            "world/latent_entropy": latentEntropy.item(),
            "world/prior_posterior_kl": priorPosteriorKL.item(),

            # Gradient health
            "gradients/world_model_norm": worldModelGradNorm.item(),

            # === SPARSE REWARD SIGNAL METRICS ===
            # Buffer composition
            "sparse_signal/event_rate_in_batch": sparse_event_rate,  # How often sparse rewards appear
            "sparse_signal/num_sparse_events": num_sparse_events,
            "sparse_signal/reward_variance": reward_variance,  # Should be > 0 for learning
            "sparse_signal/reward_min": reward_min,
            "sparse_signal/reward_max": reward_max,
            "sparse_signal/reward_range": reward_max - reward_min,  # Wider = more signal

            # Sparse reward prediction quality (CRITICAL)
            "sparse_signal/sparse_pred_error": sparse_pred_error,  # Lower = better
            "sparse_signal/sparse_pred_mean": sparse_pred_mean,
            "sparse_signal/sparse_actual_mean": sparse_actual_mean,
            "sparse_signal/sparse_sign_accuracy": sparse_sign_accuracy,  # Should be ~1.0

            # Comparison: sparse vs non-sparse prediction
            "sparse_signal/nonsparse_pred_error": nonsparse_pred_error,
            "sparse_signal/sparse_vs_nonsparse_error_ratio": sparse_pred_error / (nonsparse_pred_error + 1e-8),

            # Sparse event weighting (new)
            "sparse_signal/sparse_weight_applied": sparseWeight.item() if numSparse > 0 else 0.0,
            "sparse_signal/unweighted_reward_loss": rewardLossPerSample.mean().item(),
            "sparse_signal/weighted_reward_loss": rewardLoss.item(),
        }

        # Continue prediction metrics (if enabled)
        if self.config.useContinuationPrediction:
            metrics["world/continue_loss"] = continueLoss.item()
            continueProb = continueDist.probs
            actualContinue = 1 - data.dones[:, 1:].squeeze(-1)
            metrics["world/continue_pred_mean"] = continueProb.mean().item()
            metrics["world/continue_actual_mean"] = actualContinue.mean().item()

        return fullStates.view(-1, self.fullStateSize).detach(), metrics

    def behaviorTraining(self, fullState):
        """
        Train actor and critic in imagination.

        Args:
            fullState: Starting states from world model training (B*T, fullStateSize)

        Returns:
            metrics: Dictionary of training metrics
        """
        h, z = torch.split(fullState, (self.recurrentSize, self.latentSize), dim=-1)

        # Imagine trajectories
        fullStates, logprobs, entropies = [], [], []

        for _ in range(self.config.imaginationHorizon):
            # Get action from actor
            action, logprob, entropy = self.actor(fullState.detach(), training=True)

            # Step world model (using prior - no observation in imagination)
            h = self.recurrentModel(h, z, action)
            z, _ = self.priorNet(h)

            fullState = torch.cat((h, z), dim=-1)
            fullStates.append(fullState)
            logprobs.append(logprob)
            entropies.append(entropy)

        # Stack imagined trajectories
        fullStates = torch.stack(fullStates, dim=1)    # (B, horizon, fullStateSize)
        logprobs = torch.stack(logprobs[1:], dim=1)    # (B, horizon-1) - skip first
        entropies = torch.stack(entropies[1:], dim=1)  # (B, horizon-1)

        # Get predictions (decode from two-hot logits)
        rewardLogits = self.rewardPredictor(fullStates[:, :-1].reshape(-1, self.fullStateSize))
        rewardLogits = rewardLogits.view(fullStates.shape[0], -1, self.twoHotBins)  # (B, H-1, 255)
        predictedRewards = self.twoHot.decode(rewardLogits)  # (B, H-1)

        criticLogits = self.critic(fullStates.reshape(-1, self.fullStateSize))
        criticLogits = criticLogits.view(fullStates.shape[0], -1, self.twoHotBins)  # (B, H, 255)
        values = self.twoHot.decode(criticLogits)  # (B, H)

        if self.config.useContinuationPrediction:
            continues = self.continuePredictor(fullStates).mean
        else:
            continues = torch.full_like(predictedRewards, self.config.discount)

        # Compute lambda returns
        lambdaValues = computeLambdaValues(predictedRewards, values, continues, self.config.lambda_)

        # Normalize advantages
        _, inverseScale = self.valueMoments(lambdaValues)
        advantages = (lambdaValues - values[:, :-1]) / inverseScale

        # === Actor Loss ===
        actorLoss = -torch.mean(advantages.detach() * logprobs + self.config.entropyScale * entropies)

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        actorGradNorm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.actorOptimizer.step()

        # === Critic Loss (Two-Hot Symlog) ===
        # Get fresh logits for critic training (with gradient)
        criticLogitsForLoss = self.critic(fullStates[:, :-1].detach().reshape(-1, self.fullStateSize))
        criticLogitsForLoss = criticLogitsForLoss.view(fullStates.shape[0], -1, self.twoHotBins)  # (B, H-1, 255)
        # Two-hot cross-entropy loss with lambda returns as targets
        criticLoss = self.twoHot.loss(criticLogitsForLoss, lambdaValues.detach()).mean()

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        criticGradNorm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.criticOptimizer.step()

        # === Compute Detailed Metrics ===

        # Value normalization stats (from Moments)
        low, high = self.valueMoments.low, self.valueMoments.high

        # Collect actions for action statistics (re-sample to get actions)
        with torch.no_grad():
            sampleActions = []
            sampleState = fullState[:256]  # Sample subset to avoid memory issues
            h_sample, z_sample = torch.split(sampleState, (self.recurrentSize, self.latentSize), dim=-1)
            for _ in range(min(5, self.config.imaginationHorizon)):  # Just a few steps
                action, _, _ = self.actor(sampleState, training=True)
                sampleActions.append(action)
                h_sample = self.recurrentModel(h_sample, z_sample, action)
                z_sample, _ = self.priorNet(h_sample)
                sampleState = torch.cat((h_sample, z_sample), dim=-1)
            sampleActions = torch.cat(sampleActions, dim=0)  # (N, actionSize)

        metrics = {
            # Core losses
            "behavior/actor_loss": actorLoss.item(),
            "behavior/critic_loss": criticLoss.item(),

            # Entropy (exploration)
            "behavior/entropy_mean": entropies.mean().item(),
            "behavior/entropy_std": entropies.std().item(),
            "behavior/entropy_min": entropies.min().item(),
            "behavior/entropy_max": entropies.max().item(),

            # Log probabilities
            "behavior/logprobs_mean": logprobs.mean().item(),
            "behavior/logprobs_std": logprobs.std().item(),

            # Advantages (policy gradient signal strength)
            "behavior/advantages_mean": advantages.mean().item(),
            "behavior/advantages_std": advantages.std().item(),
            "behavior/advantages_min": advantages.min().item(),
            "behavior/advantages_max": advantages.max().item(),
            "behavior/advantages_abs_mean": advantages.abs().mean().item(),  # Signal magnitude

            # Values (critic predictions)
            "values/mean": values.mean().item(),
            "values/std": values.std().item(),
            "values/min": values.min().item(),
            "values/max": values.max().item(),

            # Lambda returns (critic targets)
            "values/lambda_returns_mean": lambdaValues.mean().item(),
            "values/lambda_returns_std": lambdaValues.std().item(),
            "values/lambda_returns_min": lambdaValues.min().item(),
            "values/lambda_returns_max": lambdaValues.max().item(),

            # Value normalization (from Moments)
            "values/norm_low": low.item(),
            "values/norm_high": high.item(),
            "values/norm_scale": (high - low).item(),

            # Imagined rewards
            "imagination/reward_mean": predictedRewards.mean().item(),
            "imagination/reward_std": predictedRewards.std().item(),
            "imagination/reward_min": predictedRewards.min().item(),
            "imagination/reward_max": predictedRewards.max().item(),

            # Imagined continues
            "imagination/continue_mean": continues.mean().item(),
            "imagination/continue_min": continues.min().item(),

            # === SPARSE SIGNAL IN IMAGINATION ===
            # Does imagination ever predict significant rewards?
            "imagination/reward_abs_mean": predictedRewards.abs().mean().item(),
            "imagination/reward_nonzero_frac": (predictedRewards.abs() > 0.1).float().mean().item(),
            "imagination/reward_significant_frac": (predictedRewards.abs() > 1.0).float().mean().item(),  # Sparse events

            # Lambda return signal strength (CRITICAL)
            "sparse_signal/lambda_return_abs_mean": lambdaValues.abs().mean().item(),
            "sparse_signal/lambda_return_nonzero_frac": (lambdaValues.abs() > 0.1).float().mean().item(),
            "sparse_signal/lambda_return_significant_frac": (lambdaValues.abs() > 1.0).float().mean().item(),

            # Value-lambda gap (how much does imagination add to values?)
            "sparse_signal/value_lambda_gap_mean": (lambdaValues - values[:, :-1]).mean().item(),
            "sparse_signal/value_lambda_gap_abs_mean": (lambdaValues - values[:, :-1]).abs().mean().item(),

            # Advantage signal (this is what drives policy learning!)
            "sparse_signal/advantage_nonzero_frac": (advantages.abs() > 0.01).float().mean().item(),
            "sparse_signal/advantage_significant_frac": (advantages.abs() > 0.1).float().mean().item(),

            # Action statistics
            "actions/mean": sampleActions.mean().item(),
            "actions/std": sampleActions.std().item(),
            "actions/min": sampleActions.min().item(),
            "actions/max": sampleActions.max().item(),
            "actions/abs_mean": sampleActions.abs().mean().item(),

            # Per-dimension action stats (for 4-dim hockey actions)
            "actions/dim0_mean": sampleActions[:, 0].mean().item(),
            "actions/dim1_mean": sampleActions[:, 1].mean().item(),
            "actions/dim2_mean": sampleActions[:, 2].mean().item(),
            "actions/dim3_mean": sampleActions[:, 3].mean().item(),

            # Gradient norms
            "gradients/actor_norm": actorGradNorm.item(),
            "gradients/critic_norm": criticGradNorm.item(),
        }

        return metrics

    @torch.no_grad()
    def act(self, observation, h=None, z=None):
        """
        Select action for a single observation.

        Args:
            observation: numpy array (obsSize,)
            h: Recurrent state (optional, for continuing episodes)
            z: Latent state (optional)

        Returns:
            action: numpy array (actionSize,)
            h: Updated recurrent state
            z: Updated latent state
        """
        if h is None:
            h = torch.zeros(1, self.recurrentSize, device=self.device)
        if z is None:
            z = torch.zeros(1, self.latentSize, device=self.device)

        # Encode observation (apply symlog first to match training)
        obs_t = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        obs_t = symlog(obs_t)  # Compress large values
        encodedObs = self.encoder(obs_t)

        # We need a dummy action for first step - use zeros
        # This matches how worldModelTraining works
        action_dummy = torch.zeros(1, self.actionSize, device=self.device)

        # Update recurrent state
        h = self.recurrentModel(h, z, action_dummy)

        # Get posterior (with observation)
        z, _ = self.posteriorNet(torch.cat((h, encodedObs), dim=-1))

        # Get action from actor
        fullState = torch.cat((h, z), dim=-1)
        action = self.actor(fullState, training=False)

        return action.cpu().numpy().reshape(-1), h, z

    def saveCheckpoint(self, checkpointPath):
        """Save agent state to file."""
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'

        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'recurrentModel': self.recurrentModel.state_dict(),
            'priorNet': self.priorNet.state_dict(),
            'posteriorNet': self.posteriorNet.state_dict(),
            'rewardPredictor': self.rewardPredictor.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'worldModelOptimizer': self.worldModelOptimizer.state_dict(),
            'actorOptimizer': self.actorOptimizer.state_dict(),
            'criticOptimizer': self.criticOptimizer.state_dict(),
            'totalEpisodes': self.totalEpisodes,
            'totalEnvSteps': self.totalEnvSteps,
            'totalGradientSteps': self.totalGradientSteps,
        }
        if self.config.useContinuationPrediction:
            checkpoint['continuePredictor'] = self.continuePredictor.state_dict()

        torch.save(checkpoint, checkpointPath)

    def loadCheckpoint(self, checkpointPath):
        """Load agent state from file."""
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(f"Checkpoint not found: {checkpointPath}")

        checkpoint = torch.load(checkpointPath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.recurrentModel.load_state_dict(checkpoint['recurrentModel'])
        self.priorNet.load_state_dict(checkpoint['priorNet'])
        self.posteriorNet.load_state_dict(checkpoint['posteriorNet'])
        self.rewardPredictor.load_state_dict(checkpoint['rewardPredictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.worldModelOptimizer.load_state_dict(checkpoint['worldModelOptimizer'])
        self.actorOptimizer.load_state_dict(checkpoint['actorOptimizer'])
        self.criticOptimizer.load_state_dict(checkpoint['criticOptimizer'])
        self.totalEpisodes = checkpoint['totalEpisodes']
        self.totalEnvSteps = checkpoint['totalEnvSteps']
        self.totalGradientSteps = checkpoint['totalGradientSteps']

        if self.config.useContinuationPrediction and 'continuePredictor' in checkpoint:
            self.continuePredictor.load_state_dict(checkpoint['continuePredictor'])
