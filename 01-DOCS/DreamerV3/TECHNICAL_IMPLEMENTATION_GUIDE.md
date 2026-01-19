Here is the distilled technical summary of the DreamerV3 algorithm, focusing strictly on implementation details, mathematical reasoning, and architectural decisions necessary to build it.

---

# DreamerV3: Technical Implementation Reference

DreamerV3 is a model-based reinforcement learning algorithm designed to work with **fixed hyperparameters** across diverse domains (from Atari pixels to continuous control). Its core innovation is replacing manual tuning with robust normalization techniques (Symlog, percentile scaling) that adapt the signals to the network, rather than tuning the network to the signals.

## 1. The "Symlog" Transformation

The critical math component that enables DreamerV3 to handle rewards of value $10^6$ and $10^{-3}$ with the same hyperparameters is the **Symlog** (Symmetric Logarithm) transformation. It compresses magnitudes while preserving sign.

## Math

Standard log is undefined for negative numbers. Symlog handles both positive and negative values symmetrically:

symlog(x)=sign(x)ln⁡(∣x∣+1)\text{symlog}(x) = \text{sign}(x) \ln(|x| + 1)**symlog**(**x**)**=**sign**(**x**)**ln**(**∣**x**∣**+**1**)**

symexp(x)=sign(x)(exp⁡(∣x∣)−1)\text{symexp}(x) = \text{sign}(x) (\exp(|x|) - 1)**symexp**(**x**)**=**sign**(**x**)**(**exp**(**∣**x**∣**)**−**1**)**

## Usage

* **Inputs:** All vector inputs (observations) are `symlog` transformed before entering the neural network.
* **Targets:** When predicting unscaled quantities (Rewards, Values), the network predicts the `symlog` transformed value. The loss is calculated in the symlog space (Symlog Squared Error).

## 2. World Model (RSSM)

The world model learns to predict the future. It is a Recurrent State-Space Model (RSSM) consisting of a deterministic path (GRU) and a stochastic path (VAE).

## Architecture

* **Encoder:** CNN (for images) or MLP (for vectors) $\rightarrow$ embeds observation $x_t$.
* **Sequence Model:** A **GRU** with  **Block-Diagonal Weights** .
  * *Implementation Detail:* Instead of a dense weight matrix, split the hidden state into blocks (e.g., 8 blocks). This reduces parameters while keeping the state size large.
* **Dynamics Predictor:** Predicts the next stochastic state $z_t$ given $h_t$.
* **Posterior (Representation):** Infers $z_t$ given $h_t$ and actual image $x_t$.

## Distributions & Stability (The "Unimix" Trick)

* **Latent State $z_t$:** A discrete categorical distribution (32 dimensions, 32 classes each).
* **Unimix:** To prevent KL collapse (where the model ignores the latent code), the distribution is strictly mixed with a uniform distribution:

  p(z)=0.99⋅pnetwork(z)+0.01⋅puniform(z)p(z) = 0.99 \cdot p_{\text{network}}(z) + 0.01 \cdot p_{\text{uniform}}(z)**p**(**z**)**=**0.99**⋅**p**network**(**z**)**+**0.01**⋅**p**uniform**(**z**)

  *Implementation Note:* Hardcode this 1% uniform mixture into your probability calculation.

## Loss Function

The objective is the Evidence Lower Bound (ELBO) with **KL Balancing** and  **Free Bits** :

Lworld=Lpred+βdynLdyn+βrepLrep\mathcal{L}_{\text{world}} = \mathcal{L}_{\text{pred}} + \beta_{\text{dyn}} \mathcal{L}_{\text{dyn}} + \beta_{\text{rep}} \mathcal{L}_{\text{rep}}**L**world**=**L**pred**+**β**dyn**L**dyn**+**β**rep**L**rep**

1. **$\mathcal{L}_{\text{pred}}$ (Prediction):** Reconstruction of image + Reward prediction + Continue flag prediction.
2. **$\mathcal{L}_{\text{dyn}}$ (Dynamics):** KL divergence ensuring the *prior* (dynamics) stays close to the  *posterior* .

   * *Weight:* 0.5 (scales the gradient to the prior).
3. **$\mathcal{L}_{\text{rep}}$ (Representation):** KL divergence ensuring the *posterior* stays close to the  *prior* .

   * *Weight:* 0.1 (scales the gradient to the posterior).
4. **Free Bits:** The KL loss is clipped below 1.0 nat. If $KL < 1.0$, the loss is 0. This prevents the model from over-optimizing the latent space for trivial details.

   max⁡(1.0,KL(p∣∣q))\max(1.0, KL(p || q))**max**(**1.0**,**K**L**(**p**∣∣**q**))**

## 3. Critic (Value Learning)

The Critic estimates the expected return. Since returns can be infinite or massive, V3 uses **Two-Hot Bucket Encoding** instead of regression.

## Two-Hot Encoding

Instead of outputting a single scalar value, the Critic outputs a softmax distribution over **exponentially spaced buckets** (e.g., 255 buckets ranging from -20 to +20 in symlog space).

**Reasoning:**

* Regression (MSE) is unstable with large outliers.
* Categorical classification (Cross Entropy) is stable but ignores the magnitude relation between classes.
* **Two-Hot** bridges this: A value $y$ falls between bucket $b_i$ and $b_{i+1}$. The target probability is assigned linearly to these two neighbors.
* *Loss:* Cross-Entropy between the network's softmax and the Two-Hot target.

## 4. Actor (Behavior Learning)

The Actor selects actions to maximize Value.

## Return Normalization

To use a fixed entropy scale (exploration) across games with score 10 vs score 1,000,000:

* Keep a running estimate of the **5th percentile** ($R_{5}$) and **95th percentile** ($R_{95}$) of returns.
* Normalize returns before computing gradients:

  Rnorm=R−R5R95−R5R_{\text{norm}} = \frac{R - R_{5}}{R_{95} - R_{5}}**R**norm**=**R**95**−**R**5**R**−**R**5
* **Important:** Scale down large returns, but *do not* scale up tiny returns (to avoid amplifying noise). Only normalize if the range $> 1$.

## Loss

Lactor=−Expectation(Rnorm)−η⋅Entropy\mathcal{L}_{\text{actor}} = -\text{Expectation}(R_{\text{norm}}) - \eta \cdot \text{Entropy}**L**actor**=**−**Expectation**(**R**norm**)**−**η**⋅**Entropy**

* $\eta = 3 \times 10^{-4}$ (Fixed for all environments).

## 5. Implementation Details "Cheat Sheet"

| Component                   | Specification                             | Reason                                                                                                                                                      |
| --------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Optimizer**         | **LaProp**(not Adam)                | Standard Adam couples momentum and learning rate scale. LaProp (RMSProp normalized gradients + Momentum) is more stable for the non-stationary world model. |
| **Gradient Clipping** | **AGC**(Adaptive Gradient Clipping) | Clips gradients based on the ratio of update norm to parameter norm. Threshold = 0.3.                                                                       |
| **Initialization**    | **Zero-Init**final layers           | Initialize the last layer of Reward and Critic networks to weight=0. This prevents massive random gradients at the start of training.                       |
| **Activations**       | **SiLU**(Swish)                     | Used everywhere instead of ReLU / ELU.                                                                                                                      |
| **Normalization**     | **LayerNorm**                       | Applied to GRU inputs and MLP layers.                                                                                                                       |
| **World Model Steps** | **Batch Size 16, Length 64**        | Trains on sequences of 64 steps.                                                                                                                            |
| **Horizon**           | **15 steps**                        | The Actor imagines 15 steps into the future for policy optimization.                                                                                        |

## The "Algorithm" Loop

1. **Interact:** Collect data with current policy, add to Replay Buffer.
2. **Train World Model:**
   * Sample sequence $(x_{1:T}, a_{1:T}, r_{1:T})$ from buffer.
   * Encode observations $x \to z$.
   * Run RSSM to get priors and posteriors.
   * Compute Reconstruction, Reward, and Continuity losses.
   * Compute KL loss (with Free Bits).
   * Update weights.
3. **Train Actor/Critic (entirely in latent imagination):**
   * Start from learned states $z_t$ (from the batch above).
   * Rollout policy for $H=15$ steps using *only* the Dynamics Predictor (no images).
   * Compute Value estimates using the Critic.
   * Update Critic using Two-Hot regression on computed targets.
   * Update Actor to maximize Critic value (normalized) + Entropy.

## Citations

* **Symlog** : [[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/29498118/8b7dbb2e-fa3e-448d-8714-78e37e7c71c5/dreamer-v3.pdf)] "We suggest the symlog squared error as a simple solution... symlog(x) = sign(x) ln(|x| + 1)"
* **Two-hot encoding** :  "The network is trained on twohot encoded targets... a generalization of onehot encoding to continuous values."[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/29498118/8b7dbb2e-fa3e-448d-8714-78e37e7c71c5/dreamer-v3.pdf)]
* **Return Normalization** :  "we normalize returns to be approximately contained in the interval... compute the range from the 5th to the 95th return percentile."[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/29498118/8b7dbb2e-fa3e-448d-8714-78e37e7c71c5/dreamer-v3.pdf)]
* **Free Bits** :  "we employ free bits by clipping the dynamics and representation losses below the value of 1 nat."[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/29498118/8b7dbb2e-fa3e-448d-8714-78e37e7c71c5/dreamer-v3.pdf)]
