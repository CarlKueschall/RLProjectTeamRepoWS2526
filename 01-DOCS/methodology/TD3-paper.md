Based on the provided paper, here is the structured Markdown representation focusing on the core technical content, mathematical formulations, and tables.

# Addressing Function Approximation Error in Actor-Critic Methods

**Authors:** Scott Fujimoto, Herke van Hoof, David Meger

**Published:** ICML 2018[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/29498118/2ea00101-54cc-49f4-a2c0-ab4b89affb86/1802.09477v3.pdf)]

## Abstract

In value-based reinforcement learning, function approximation errors lead to overestimated value estimates and suboptimal policies. This paper shows that this overestimation bias persists in actor-critic settings. The authors propose the **Twin Delayed Deep Deterministic policy gradient (TD3)** algorithm, which uses three primary mechanisms: Clipped Double Q-learning, delayed policy updates, and target policy smoothing.[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/29498118/2ea00101-54cc-49f4-a2c0-ab4b89affb86/1802.09477v3.pdf)]

---

## 1. Background

In actor-critic methods, the objective is to find the optimal policy πϕ\pi_{\phi}**π**ϕ that maximizes the expected return J(ϕ)=Esi∼ρπ,ai∼π[R0]J(\phi) = \mathbb{E}_{s_i \sim \rho^\pi, a_i \sim \pi} [R_0]**J**(**ϕ**)**=**E**s**i**∼**ρ**π**,**a**i**∼**π**[**R**0**]. The policy (actor) is updated via the deterministic policy gradient:

∇ϕJ(ϕ)=Es∼ρπ[∇aQπ(s,a)∣a=π(s)∇ϕπϕ(s)](1)\nabla_{\phi} J(\phi) = \mathbb{E}_{s \sim \rho^\pi} [\nabla_a Q^\pi(s, a)|_{a=\pi(s)} \nabla_{\phi} \pi_{\phi}(s)] \quad (1)**∇**ϕ**J**(**ϕ**)**=**E**s**∼**ρ**π**[**∇**a**Q**π**(**s**,**a**)**∣**a**=**π**(**s**)**∇**ϕ**π**ϕ**(**s**)]**(**1**)**

The critic Qπ(s,a)Q^\pi(s, a)**Q**π**(**s**,**a**)** is updated using temporal difference (TD) learning based on the Bellman equation:

Qπ(s,a)=r+γEs′,a′[Qπ(s′,a′)](2)Q^\pi(s, a) = r + \gamma \mathbb{E}_{s', a'} [Q^\pi(s', a')] \quad (2)**Q**π**(**s**,**a**)**=**r**+**γ**E**s**′**,**a**′**[**Q**π**(**s**′**,**a**′**)]**(**2**)

In deep Q-learning, the target yy**y** is maintained using a secondary frozen target network Qθ′Q_{\theta'}**Q**θ**′**:

y=r+γQθ′(s′,πϕ′(s′))(3)y = r + \gamma Q_{\theta'}(s', \pi_{\phi'}(s')) \quad (3)**y**=**r**+**γ**Q**θ**′**(**s**′**,**π**ϕ**′**(**s**′**))**(**3**)

---

## 2. Overestimation Bias

The paper proves that value estimates in deterministic policy gradients suffer from overestimation bias due to the maximization of noisy estimates.

## 2.1 Clipped Double Q-Learning

To address overestimation, the authors propose a clipped variant of Double Q-learning. Instead of maintaining two actors, they use a single actor optimized with respect to Q1Q_1**Q**1 and take the minimum of two critics for the target calculation:

y1=r+γmin⁡i=1,2Qθi′(s′,πϕ(s′))(10)y_1 = r + \gamma \min_{i=1,2} Q_{\theta_i'}(s', \pi_{\phi}(s')) \quad (10)**y**1**=**r**+**γ**min**i**=**1**,**2**Q**θ**i**′**(**s**′**,**π**ϕ**(**s**′**))**(**10**)**

This underestimation bias is preferable to overestimation, as underestimated values are not explicitly propagated by the policy update.[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/29498118/2ea00101-54cc-49f4-a2c0-ab4b89affb86/1802.09477v3.pdf)]

---

## 3. Addressing Variance

High variance in value estimates provides noisy gradients, reducing learning speed and performance.

## 3.1 Target Networks and Delayed Policy Updates

The authors suggest that policy updates should be performed at a lower frequency than value updates to allow the critic to converge. The actor πϕ\pi_{\phi}**π**ϕ and target networks are updated every dd**d** iterations:

* Update θi\theta_i**θ**i by minimizing TD error.
* Every dd**d** steps, update ϕ\phi**ϕ** by following the policy gradient ∇ϕJ(ϕ)\nabla_{\phi} J(\phi)**∇**ϕ**J**(**ϕ**).

## 3.2 Target Policy Smoothing

To reduce overfitting to narrow peaks in the value estimate, a SARSA-style regularization adds small random noise to the target actions:

y=r+γQθ′(s′,πϕ′(s′)+ϵ),ϵ∼clip(N(0,σ),−c,c)(14)y = r + \gamma Q_{\theta'}(s', \pi_{\phi'}(s') + \epsilon), \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c) \quad (14)**y**=**r**+**γ**Q**θ**′**(**s**′**,**π**ϕ**′**(**s**′**)**+**ϵ**)**,**ϵ**∼**clip**(**N**(**0**,**σ**)**,**−**c**,**c**)**(**14**)

---

## 4. TD3 Algorithm

The complete algorithm combines these components as follows:

| Step        | Action                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | Initialize criticsQθ1,Qθ2Q_{\theta_1}, Q_{\theta_2}**Q**θ**1**,**Q**θ**2**and actorπϕ\pi_{\phi}**π**ϕwith random parameters.                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **2** | Initialize target networksθ1′←θ1,θ2′←θ2,ϕ′←ϕ\theta_1' \leftarrow \theta_1, \theta_2' \leftarrow \theta_2, \phi' \leftarrow \phi**θ**1**′**←**θ**1**,**θ**2**′**←**θ**2**,**ϕ**′**←**ϕ.                                                                                                                                                                                                                                                                                                                           |
| **3** | Select actiona=πϕ(s)+ϵa = \pi_{\phi}(s) + \epsilon**a**=**π**ϕ**(**s**)**+**ϵ**whereϵ∼N(0,σ)\epsilon \sim \mathcal{N}(0, \sigma)**ϵ**∼**N**(**0**,**σ**).                                                                                                                                                                                                                                                                                                                                                                                                        |
| **4** | Compute target actions with noise:a~=πϕ′(s′)+ϵ,ϵ∼clip(N(0,σ~),−c,c)\tilde{a} = \pi_{\phi'}(s') + \epsilon, \epsilon \sim \text{clip}(\mathcal{N}(0, \tilde{\sigma}), -c, c)**a**~**=**π**ϕ**′**(**s**′**)**+**ϵ**,**ϵ**∼**clip**(**N**(**0**,**σ**~**)**,**−**c**,**c**).[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/29498118/2ea00101-54cc-49f4-a2c0-ab4b89affb86/1802.09477v3.pdf)]****                                                                                                     |
| **5** | Set targety=r+γmin⁡i=1,2Qθi′(s′,a~)y = r + \gamma \min_{i=1,2} Q_{\theta_i'}(s', \tilde{a})**y**=**r**+**γ**min**i**=**1**,**2**Q**θ**i**′**(**s**′**,**a**~**).                                                                                                                                                                                                                                                                                                                                                                                                          |
| **6** | Update criticsθi\theta_i**θ**ivia gradient descent on(y−Qθi(s,a))2(y - Q_{\theta_i}(s, a))^2**(**y**−**Q**θ**i**(**s**,**a**)**)**2**.                                                                                                                                                                                                                                                                                                                                                                                             |
| **7** | **Everydd**d**steps:**Updateϕ\phi**ϕ**using the deterministic policy gradient.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **8** | **Everydd**d**steps:**Update targets:θi′←τθi+(1−τ)θi′\theta_i' \leftarrow \tau \theta_i + (1-\tau)\theta_i'**θ**i**′**←**τ**θ**i**+**(**1**−**τ**)**θ**i**′andϕ′←τϕ+(1−τ)ϕ′\phi' \leftarrow \tau \phi + (1-\tau)\phi'**ϕ**′**←**τ**ϕ**+**(**1**−**τ**)**ϕ**′**.[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/29498118/2ea00101-54cc-49f4-a2c0-ab4b89affb86/1802.09477v3.pdf)] |

---

## 5. Experimental Hyperparameters

The following table compares the hyperparameters used in TD3 versus the OpenAI Baselines DDPG implementation.

| Hyperparameter                          | TD3 (Ours)                                                        | DDPG (Baseline)                                                                                   |
| --------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Critic Learning Rate                    | 10−310^{-3}**1**0**−**3                                   | 10−310^{-3}**1**0**−**3                                                                   |
| Actor Learning Rate                     | 10−310^{-3}**1**0**−**3                                   | 10−410^{-4}**1**0**−**4                                                                   |
| Target Update Rate (τ\tau**τ**) | 5⋅10−35 \cdot 10^{-3}**5**⋅**1**0**−**3           | 10−310^{-3}**1**0**−**3                                                                   |
| Batch Size                              | 100                                                               | 64                                                                                                |
| Optimizer                               | Adam                                                              | Adam                                                                                              |
| Exploration Policy                      | N(0,0.1)\mathcal{N}(0, 0.1)**N**(**0**,**0.1**) | OU Noise (θ=0.15,σ=0.2\theta=0.15, \sigma=0.2**θ**=**0.15**,**σ**=**0.2**)                    |
| Normalized Observations                 | False                                                             | True                                                                                              |
| Critic Regularization                   | None                                                              | 10−2⋅∥θ∥210^{-2} \cdot\|\theta\|^2**1**0**−**2**⋅**∥**θ**∥**2** |

---

## 6. Appendix: Convergence Proof

The paper provides a convergence proof for Clipped Double Q-learning in finite MDPs. The update rule for the difference between two critics ΔBA=QB−QA\Delta_{BA} = Q_B - Q_A**Δ**B**A**=**Q**B**−**Q**A** follows:

ΔBAt+1(s,a)=(1−αt(s,a))ΔBAt(s,a)(21)\Delta_{BA}^{t+1}(s, a) = (1 - \alpha_t(s, a))\Delta_{BA}^t(s, a) \quad (21)**Δ**B**A**t**+**1**(**s**,**a**)**=**(**1**−**α**t**(**s**,**a**))**Δ**B**A**t**(**s**,**a**)**(**21**)

This shows that ΔBA\Delta_{BA}**Δ**B**A** converges to 0 with probability 1, implying that both critics converge to the optimal value function Q∗Q^***Q**∗.[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/29498118/2ea00101-54cc-49f4-a2c0-ab4b89affb86/1802.09477v3.pdf)]
