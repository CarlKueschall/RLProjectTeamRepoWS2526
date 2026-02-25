# What We Learned

## 1. DreamSmooth is Essential for Sparse Rewards

Without temporal reward propagation, the agent peaks early (95% at ~105k steps) then drifts down (88–92%). DreamSmooth propagates goal rewards backward through time, giving a denser learning signal. With it, performance rises to 99% and stays there.

**Takeaway:** For environments with rewards only at episode boundaries, DreamSmooth (or similar temporal credit assignment) is not optional.

## 2. Two-Hot Symlog Helps but Gaussian Can Work

Theory says MSE fails on sparse {−10, 0, +10} distributions. In practice, Gaussian reached 97%—no collapse. Two-Hot reached 98% with slightly better final reward. The gap is small but consistent.

**Takeaway:** Two-Hot is the safer, theoretically grounded choice. Gaussian is acceptable if you want to avoid the extra implementation.

## 3. Early Learning Speed ≠ Final Performance

DreamSmooth OFF learned faster early but did not end up best. DreamSmooth ON learned slower initially, then overtook and stayed ahead. Stability over time matters more than early speed.

## 4. Run Length and Stability Correlate with Modifications

DreamSmooth ON ran to 393k steps; OFF stopped at 288k. Better credit assignment appears to support longer, more stable training.
