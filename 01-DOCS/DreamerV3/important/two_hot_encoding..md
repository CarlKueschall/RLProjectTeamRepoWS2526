Here is the complete implementation guide for the **Symlog Two-Hot Encoding** used in DreamerV3 (Hafner et al., 2023).

### 1. Mathematical Formulation

The Two-Hot encoding converts a scalar $x$ into a categorical distribution over fixed bins.

**Step 1: Symlog Transformation**
First, compress the continuous value $x$ (which could be $10^5$ or $10^{-5}$) into a symmetric logarithmic space.
\[ \text{symlog}(x) = \text{sign}(x) \ln(|x| + 1) \]

**Step 2: Discretization (Two-Hot)**
We have $B=255$ bins equally spaced in the *symlog* domain between $[-20, 20]$.
Let the bin edges be $b_1, \dots, b_B$.
Given a symlog-transformed value $y = \text{symlog}(x)$, find the index $k$ such that $b_k \leq y < b_{k+1}$.
The target probability distribution has only two non-zero entries at indices $k$ and $k+1$:
\[ p_k = \frac{|b_{k+1} - y|}{|b_{k+1} - b_k|} \quad (\text{weight for lower bin}) \]
\[ p_{k+1} = \frac{|y - b_k|}{|b_{k+1} - b_k|} \quad (\text{weight for upper bin}) \]
Basically, if $y$ is 30% of the way from $b_k$ to $b_{k+1}$, then bin $k+1$ gets 0.3 probability and bin $k$ gets 0.7 probability.

**Step 3: Prediction to Scalar**
To convert the network's predicted softmax distribution $p(z)$ back to a scalar value:
\[ \hat{y} = \sum_{i=1}^{B} p_i b_i \]
\[ \text{Value} = \text{symexp}(\hat{y}) = \text{sign}(\hat{y}) (\exp(|\hat{y}|) - 1) \]

### 2. Bin Configuration

* **Count:** 255 bins.
* **Range:** $[-20, +20]$ in Symlog space.
  * *Note:* $\text{symexp}(20) \approx 4.8 \times 10^8$. This covers effectively infinite rewards.
* **Spacing:** Linearly spaced *after* the symlog transformation (which means exponentially spaced in real value).

### 3. Loss Function

The network outputs logits for the 255 classes.

* **Loss:** Categorical Cross-Entropy between the predicted logits and the calculated Two-Hot target distribution.
  \[ \mathcal{L} = - \sum_{i=1}^{B} p_{\text{target}, i} \log(p_{\text{pred}, i}) \]
  *Since $p_{\text{target}}$ is zero everywhere except indices $k$ and $k+1$, this simplifies to just the weighted log-prob of the two neighbors.*

### 4. Implementation Code (PyTorch)

Based on Danijar Hafner's JAX logic, adapted to PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoHotSymlog(nn.Module):
    def __init__(self, bins=255, min_val=-20.0, max_val=20.0):
        super().__init__()
        self.bins = bins
        # Create bins: [-20, ..., +20]
        self.register_buffer("bin_centers", torch.linspace(min_val, max_val, bins))
        # Bin step size (assuming linear spacing in symlog space)
        self.step = (max_val - min_val) / (bins - 1)

    def symlog(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    def symexp(self, x):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

    def forward_loss(self, logits, target_scalar):
        """
        logits: [batch, bins]
        target_scalar: [batch, 1] real values (rewards/values)
        """
        # 1. Transform target to symlog space
        y = self.symlog(target_scalar)
      
        # 2. Find bucket indices
        # Clamp target to range of bins to handle outliers
        y = torch.clamp(y, self.bin_centers[0], self.bin_centers[-1])
      
        # Calculate continuous index (0 to 254)
        below_idx = (y - self.bin_centers[0]) / self.step
        k = below_idx.long()             # Lower bin index
        k_plus_1 = k + 1                 # Upper bin index
      
        # Clamp indices to ensure they are valid
        k = torch.clamp(k, 0, self.bins - 1)
        k_plus_1 = torch.clamp(k_plus_1, 0, self.bins - 1)
      
        # 3. Calculate weights (alpha is distance from lower bin)
        alpha = below_idx - k.float()
      
        # 4. Create two-hot target distribution
        # In practice, we just use Cross Entropy with soft labels.
        # But explicitly: target_probs[k] = 1-alpha, target_probs[k+1] = alpha
      
        # Efficient calculation using gather/scatter is complex, 
        # simpler is to construct the sparse target or use custom loss.
        # Here is a direct sparse cross-entropy implementation:
      
        log_probs = F.log_softmax(logits, dim=-1)
      
        # Gather log probs for the two bins
        log_p_k = log_probs.gather(1, k.view(-1, 1)).squeeze(-1)
        log_p_k1 = log_probs.gather(1, k_plus_1.view(-1, 1)).squeeze(-1)
      
        # Cross Entropy = - sum( p_target * log_p_pred )
        loss = - ( (1 - alpha) * log_p_k + alpha * log_p_k1 )
      
        return loss.mean()

    def decode(self, logits):
        """
        Convert network logits back to scalar value.
        """
        probs = F.softmax(logits, dim=-1)
        # Expected value in symlog space
        y_hat = torch.sum(probs * self.bin_centers, dim=-1)
        # Inverse transform
        return self.symexp(y_hat)
```

### 5. Application in DreamerV3

The two-hot encoding is used for **Both**:

1. **Reward Predictor:** The World Model predicts rewards $r_t$ using this method.
2. **Critic (Value Function):** The Critic predicts value $V(s_t)$ using this method.
   * *Note:* The Critic output layer is usually initialized to **zero weights** so the initial prediction is a uniform distribution (value $\approx 0$).

### 6. Edge Cases

* **Out of Range:** If a value exceeds `symexp(20)` (approx 485 million), it is **clipped** to the min/max bin index. The loss will just push the probability mass into the final bin (index 254), effectively telling the network "it's at least this big."
* **Precision:** Because the bins are in symlog space, the resolution is fine near zero (discriminating 0.1 from 0.2) and coarse near millions (discriminating 1,000,000 from 1,100,000). This perfectly matches RL requirements where small rewards matter for micro-adjustments, but the difference between massive returns matters less.

### Summary Checklist

1. **Transform:** $y = \text{symlog}(x)$
2. **Bin:** $y \to$ indices $k, k+1$ in $[-20, 20]$
3. **Target:** Weight $1-\alpha$ to $k$, $\alpha$ to $k+1$
4. **Loss:** CrossEntropy(logits, target_probs)
5. **Decode:** $\text{symexp}(\sum p_i b_i)$

### References

* **Paper:** "Mastering Diverse Domains through World Models" (Hafner et al., 2023). See **Equation 8, 9, 10, 11**.
* **Appendix B:** Contains the exact hyperparams (Bins=255).
* **Source:** `tools.py` in the official repo `danijar/dreamerv3` (JAX) or `NM512/dreamerv3-torch` (PyTorch).
