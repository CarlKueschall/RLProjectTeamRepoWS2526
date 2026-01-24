# Environment Mechanics: Puck Holding & Action Processing

## Source Location
`/opt/homebrew/Caskroom/miniconda/base/envs/py310/lib/python3.10/site-packages/hockey/hockey_env.py`

## Action Space (keep_mode=True)

4 dimensions per player, [-1, +1] each:

| Dim | Controls | Force Applied |
|-----|----------|---------------|
| 0 | X-translation | action * FORCEMULTIPLIER (6000) |
| 1 | Y-translation | action * FORCEMULTIPLIER (6000) |
| 2 | Rotation | action * TORQUEMULTIPLIER (400) |
| 3 | **SHOOT** | Binary trigger: fires if > 0.5 |

**Our agent**: abs_mean=0.98 means ~5880 Newtons force applied every timestep.

## Puck Acquisition Mechanism

```python
# ContactDetector.BeginContact() - line 65
if self.env.keep_mode and self.env.puck.linearVelocity[0] < 0.1:
    if self.env.player1_has_puck == 0:
        self.env.player1_has_puck = MAX_TIME_KEEP_PUCK  # = 15 frames
```

**CRITICAL**: Puck holding triggers when:
1. Player makes physical contact with puck, AND
2. **Puck X-velocity < 0.1** (nearly stationary in X)

This is NOT an action threshold - it's a PUCK VELOCITY threshold.

## Shooting Mechanism

```python
# step() - line 668-680
if self.player1_has_puck > 1:
    self._keep_puck(self.player1)       # Puck follows player
    self.player1_has_puck -= 1          # Timer counts down
    if self.player1_has_puck == 1 or action[3] > 0.5:  # SHOOT CHECK
        self._shoot(self.player1, True) # Apply SHOOTFORCEMULTIPLIER=60
        self.player1_has_puck = 0       # Release
```

## Why Our Agent Never Holds the Puck

**Chain of failure:**

1. Agent outputs action[0:2] ~ 0.98 magnitude = 5880N force
2. When agent contacts puck, the collision imparts massive velocity
3. Puck velocity >> 0.1 at the moment of contact
4. Acquisition condition NEVER satisfied
5. Even if it were, action[3] ~ 0.98 > 0.5 means instant shoot

**What the strong bot does differently:**
```python
# BasicOpponent.act() uses PD control with kp=10
# This produces MODERATE actions that approach the puck gently
# Then checks: if obs[16] > 0 and obs[16] < 7: shoot = 1.0
# Otherwise shoot = 0.0 (holds the puck strategically)
```

## Key Constants

```python
FPS = 50
MAX_TIME_KEEP_PUCK = 15       # 0.3 seconds hold time
FORCEMULTIPLIER = 6000         # Translation force scaling
TORQUEMULTIPLIER = 400         # Rotation torque scaling
SHOOTFORCEMULTIPLIER = 60      # Shot force (much smaller!)
MAX_PUCK_SPEED = 25            # Hard cap on puck speed
```

## Observation Space (keep_mode=True, 18-dim)

| Index | Content |
|-------|---------|
| 0-1 | Player position (x, y) |
| 2 | Player angle |
| 3-4 | Player velocity (vx, vy) |
| 5 | Player angular velocity |
| 6-7 | Opponent position (x, y) |
| 8 | Opponent angle |
| 9-10 | Opponent velocity (vx, vy) |
| 11 | Opponent angular velocity |
| 12-13 | Puck position (x, y) |
| 14-15 | Puck velocity (vx, vy) |
| **16** | **Player 1 puck-hold timer** |
| **17** | **Player 2 puck-hold timer** |

## Implications for DreamerV3

The world model sees obs[16] (puck-hold timer) but the agent never triggers it because it never meets the acquisition condition. The world model likely learns that this dimension is always 0, and the actor never learns behaviors that would make it non-zero.
