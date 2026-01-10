# Deep RL Design for Hockey-Env Self-Play (TD3, 100k+ Episodes)

**Main takeaways (actionable):**

* **Ties:** Treat draws as *mildly worse than continuing* but  *much better than a loss* . For a terminal scale of win =+1=+1**=**+**1**, loss =−1=-1**=**−**1**, use **tie in [−0.3,−0.6][-0.3,-0.6]**[**−**0.3**,**−**0.6**]**** plus **time‑dependent pressure** (small negative per step in a 0–0 game) and **goal‑differential shaping** rather than a huge constant tie penalty. This is consistent with how competitive RL and games handle win/draw/loss and avoids pathological over‑aggression.**stanford**+2****youtube****
* **Opponents & replay:** Use **population-based self‑play** with **PFSP-style opponent sampling** (AlphaStar style) and **blocks of episodes vs the same opponent (≈10–30)** before switching, rather than switching every 1–3 episodes. Keep a **stable “anchor” (basic) opponent fraction (≈50–70%)** and  **separate or at least tagged replay for anchor vs self‑play opponents** , then **stratified sampling** that enforces a minimum share of anchor data.**googleapis**+5
* **100k‑episode scaling:** For ~25M transitions:
  * Use **LR decay** (step or cosine) over training instead of constant LR.
  * Grow replay to **1–2M transitions** (≈4–8% of total) rather than 500k, and keep  **sample reuse near 1×** .
  * Keep **batch size large (512–1024)** and constant; large batches are stable for TD3‑style methods.**ifaamas**+2
  * Use **warmup ~2–5% of total steps** (random policy / high exploration) rather than 0.5%.
  * Expand the **self‑play pool to ~20–30 checkpoints** and use PFSP over that pool.**arxiv**+2
  * Evaluate **more frequently early (every ~500 episodes)** and  **less frequently later (every 2–5k)** , using a **TrueSkill ladder against a fixed reference set** rather than single-opponent winrates.**openreview**+5

Below, each challenge is treated separately with literature, concrete numbers, code patterns, impact estimates, and caveats.

---

## Context from Hockey / Laser-Hockey Work

Several groups have already solved **laser-hockey/hockey-env** variants for the Tübingen RL challenge using DDPG, SAC, PPO, and TD3:

* **anticdimi/laser-hockey** : winning entry with Dueling DQN + PER, SAC, and DDPG; environment uses  **+10/−10/0 terminal reward for win/loss/draw plus dense shaping for puck proximity** , and games are declared draw after 230 steps.**github**+1
* **meier-johannes94/Reinforcement-Learning-PPO** : PPO agent with self‑play against basic opponent + historical checkpoints; same reward scheme (+10/−10/0, 0 for draw, dense “distance to ball” shaping).[github](https://github.com/meier-johannes94/Reinforcement-Learning-PPO)
* **the-klingspor/laser-hockey** : 2nd and 3rd place agents (strongest model‑free agents, only beaten by a MuZero‑based approach) with an extensive report in `/report` describing training details.[github](https://github.com/the-klingspor/laser-hockey)
* **kitteltom/rl-laser-hockey** : TD3 agent explicitly “solving” laser‑hockey.[github](https://github.com/kitteltom/rl-laser-hockey)
* **Robot Air Hockey Challenge** (DreamerV3 + self-play & opponent mixtures) uses a similar air‑hockey setting and emphasizes  **opponent diversity (aggressive vs defensive) early in training** .**ias.informatik.tu-darmstadt**+1

These works generally:

* **Do not change draw = 0 at terminal** , but
* Use **dense positional shaping** and
* **Opponent pools or self‑play** (historical checkpoints + basic opponent) to avoid overfitting.**github**+2

Your issues (very high tie rate, unstable targets, 100k‑episode scale) are not fully addressed there, but they give good baselines and show that **reward tweaks and opponent scheduling** are where you can diverge.

---

## CHALLENGE 1 – Excessive Tie Rate / Overly Defensive Policy

## 1.1. What the literature actually does with draws and reward shaping

**Board & discrete games:**

* Sutton & Barto’s canonical Tic‑Tac‑Toe setup uses  **+1 for win, −1 for loss, 0 for draw** . This is “neutral” and tends to produce **risk‑neutral** agents; not what you want.[stanford](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
* Many implementations use a *non‑neutral draw* to affect playstyle:
  * A tutorial Tic‑Tac‑Toe agent gives **1 (win), 0 (loss), 0.5 (draw)** to reflect that a draw is “half as good as a win”.**youtube**
  * A minimal C RL implementation for Tic‑Tac‑Toe uses  **1 (win), −1 (loss), +0.2 (draw)** :[github](https://github.com/antirez/ttt-rl)

    <pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-normal bg-subtler"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden rounded-full border-subtlest ring-subtlest divide-subtlest bg-base"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtler"><button data-testid="copy-code-button" aria-label="Copy code" type="button" class="focus-visible:bg-subtle hover:bg-subtle text-quiet  hover:text-foreground dark:hover:bg-subtle font-sans focus:outline-none outline-none outline-transparent transition duration-300 ease-out select-none items-center relative group/button font-semimedium justify-center text-center items-center rounded-full cursor-pointer active:scale-[0.97] active:duration-150 active:ease-outExpo origin-center whitespace-nowrap inline-flex text-sm h-8 aspect-square" data-state="closed"><div class="flex items-center min-w-0 gap-two justify-center"><div class="flex shrink-0 items-center justify-center size-4"><svg role="img" class="inline-flex fill-current" width="16" height="16"><use xlink:href="#pplx-icon-copy"></use></svg></div></div></button></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-subtle py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">c</div></div><div><span><code><span><span class="token token">if</span><span></span><span class="token token punctuation">(</span><span>winner </span><span class="token token operator">==</span><span></span><span class="token token char">'T'</span><span class="token token punctuation">)</span><span>      reward </span><span class="token token operator">=</span><span></span><span class="token token">0.2f</span><span class="token token punctuation">;</span><span></span><span class="token token">// Small reward for draw</span><span>
    </span></span><span><span></span><span class="token token">else</span><span></span><span class="token token">if</span><span></span><span class="token token punctuation">(</span><span>winner </span><span class="token token operator">==</span><span> me</span><span class="token token punctuation">)</span><span>  reward </span><span class="token token operator">=</span><span></span><span class="token token">1.0f</span><span class="token token punctuation">;</span><span></span><span class="token token">// Win</span><span>
    </span></span><span><span></span><span class="token token">else</span><span>                    reward </span><span class="token token operator">=</span><span></span><span class="token token operator">-</span><span class="token token">1.0f</span><span class="token token punctuation">;</span><span></span><span class="token token">// Loss</span><span>
    </span></span><span></span></code></span></div></div></div></pre>

    This makes the agent **actively avoid losses** but be content with draws, causing *defensive* play.
  * A game‑theory example in a modified Rock–Paper–Scissors uses **1 (win), −1 (loss), −2 (draw)** so “it is in both players’ interests to avoid draws entirely”. This is an extreme **draw‑penalty** that heavily disincentivizes stalemates.[kth.diva-portal](https://kth.diva-portal.org/smash/get/diva2:1795583/FULLTEXT01.pdf)
* These examples show that  **there is no universal numeric rule** ; the draw reward is tuned to control **risk appetite** and strategy balance.

**Sports / football RL:**

* Google Research Football and follow‑ups often use **±1 for scoring/conceding** plus dense shaping for field position, possession, shot creation, etc., and treat non‑terminal timesteps with shaped rewards while terminal results reflect goal difference.**fse.studenttheses.rug**+1
* A recent thesis on **dense reward shaping in football** designs a reward:
  * ±1 for scoring/conceding (“Scoring” reward),
  * +0.1 per field “checkpoint” advanced with ball (up to +1 total),
  * Smaller positional and action-based rewards (e.g., starting/ending position density rewards up to about 0.001 relative to goal).[fse.studenttheses.rug](https://fse.studenttheses.ub.rug.nl/33947/1/bAI2024vanDommeleA.pdf)
  * Their conclusion: **dense shaping plus sparse scoring** significantly improves learning and allows encoding “aggressive play” preferences.

**Multi-agent reward shaping theory:**

* **Potential-based Reward Shaping (PBRS)** : add a term F(s,s′)=γΦ(s′)−Φ(s)F(s,s') = \gamma \Phi(s') - \Phi(s)**F**(**s**,**s**′**)**=**γ**Φ**(**s**′**)**−**Φ**(**s**)** to the reward; this speeds learning but  **does not change the optimal policy / Nash equilibria** .**arxiv**+1****
* **DRiP (Difference Rewards + PBRS)** and **dynamic PBRS** extend this idea to multi-agent settings, keeping equilibrium structure while changing learning dynamics.**ifaamas**+1
* These frameworks support **time- and state-dependent shaping** (e.g., increasing penalty as the game drags on) *without* changing which policies are optimal in the underlying game, so long as they remain potential-based.**arxiv**+1

**Aggressive vs defensive balance in competitive RL:**

* The **FightLadder** benchmark for competitive MARL notes that **human players naturally adopt “defensive counterattack” strategies** that exploit overly aggressive RL agents; conversely, RL agents often become *too cautious* if reward structures make “not losing” nearly as good as winning.[sites.google](https://sites.google.com/view/fightladder/home)
* The football thesis explicitly notes that sparse “win/loss only” rewards lead to  **overly conservative, brittle policies** , and that **dense action‑ and position‑based shaping is essential** to get agents to learn nuanced attacking behavior.[fse.studenttheses.rug](https://fse.studenttheses.ub.rug.nl/33947/1/bAI2024vanDommeleA.pdf)

**Hockey / air‑hockey specific:**

* The Tübingen laser‑hockey challenge environments use **+10 / −10 / 0 for win/loss/draw** and dense “distance to ball” shaping per step.[github](https://github.com/meier-johannes94/Reinforcement-Learning-PPO)
* A model‑based deep RL air‑hockey solution introduces a  **pool of strong aggressive and defensive opponents** , so “the new agent is immediately exposed to a diverse pool of advanced opponents”, where “aggressive opponents immediately incentivize the new agent to defend against their risky behaviour, while the defensive opponents provide challenging agents to score goals against”.**github**+1
* None of these works add an explicit  *draw penalty* , but their **shaping and opponent mix** make persistent draws rare against learned agents.

## 1.2. Concrete recommendations for tie penalties and shaping

Given your situation (agent defends perfectly, 44% ties vs weak opponent, 100% win when decisive), you need to  **make a draw substantially worse than “continuing to play” but still clearly better than a loss** .

Assume **win = +1, loss = −1** as your terminal rewards (you can scale by 10 if you prefer). Reasonable numbers, guided by the examples above:

* **Tie penalty magnitude:**
  * **Mild:** tie =−0.3= -0.3**=**−**0.3** (30% of loss magnitude) – encourages some risk‑taking, still fairly defensive.
  * **Moderate:** tie =−0.5= -0.5**=**−**0.5** – roughly “half a loss”; strongly incentivises breaking stalemates.
  * **Aggressive:** tie =−0.7= -0.7**=**−**0.7** or below – might over‑encourage reckless all‑in attacks.
* **Suggested range for your case:** **tie ≈−0.3≈ -0.3**≈**−**0.3** to −0.5-0.5**−**0.5** (if win =+1= +1**=**+**1**, loss =−1= -1**=**−**1**), mirroring the “small reward for draw” 0.2 in Tic‑Tac‑Toe but inverted to encourage risk instead of safety.**github**+1****

This can be written as a simple rule:

rterminal={+1if win−1if loss−αif drawwith α∈[0.3,0.5].r_{\text{terminal}} =
\begin{cases}
+1 & \text{if win} \\
-1 & \text{if loss} \\
-\alpha & \text{if draw}
\end{cases}
\quad\text{with }\alpha \in [0.3,0.5].**r**terminal**=**⎩⎨⎧**+**1**−**1**−**α**if win**if loss**if draw**with **α**∈**[**0.3**,**0.5**]**.
If your environment currently uses ±10/0 like the original challenge, use  **win = +10, loss = −10, draw = −3 to −5** .[github](https://github.com/meier-johannes94/Reinforcement-Learning-PPO)

## Time‑adaptive tie penalties

Dynamic shaping literature (dynamic PBRS) supports  **time‑varying potentials** . For hockey‑env, a natural potential is  **“negative remaining time without a goal”** . Two practical options:[arxiv](https://arxiv.org/html/2408.10215v1)

1. **Per‑step stalemate penalty after some threshold T0T_0**T**0:**
   * For each step t>T0t > T_0**t**>**T**0 where score is tied and puck is not in an obviously dangerous zone, add:

     rtstalemate=−β,β∈[0.001,0.01]r_t^{\text{stalemate}} = -\beta, \quad \beta \in [0.001, 0.01]**r**t**stalemate**=**−**β**,**β**∈**[**0.001**,**0.01**]
   * Over a 230‑step max episode as in the challenge, with T0=150T_0 = 150**T**0**=**150, this yields an added penalty up to about −0.8-0.8**−**0.8 to −8-8**−**8 depending on scale, shared by both agents.[github](https://github.com/meier-johannes94/Reinforcement-Learning-PPO)
   * This **gradually increases pressure to “do something”** instead of happily blocking.
2. **Terminal draw penalty proportional to elapsed time:**
   * At a draw, compute:

     rdraw=−α⋅TepisodeTmax⁡r_{\text{draw}} = -\alpha \cdot \frac{T_{\text{episode}}}{T_{\max}}**r**draw**=**−**α**⋅**T**m**a**x**T**episode
   * So a fast draw (rare in hockey) is mildly penalized (close to 0), and a full-length stalemate is penalized by −α-\alpha**−**α.

Both are easily implemented as PBRS‑friendly shaping if you treat **time** as part of the potential Φ(s)\Phi(s)**Φ**(**s**).[arxiv](https://arxiv.org/html/2408.10215v1)

## Goal‑differential and dense shaping

To avoid hacking the game by simply over‑penalizing draws, combine draw penalties with **dense, symmetric shaping** that explicitly rewards controlled aggression:

* **Goal differential at terminal:**

  rterminal=λ1⋅sign(Δgoals)r_{\text{terminal}} =
  \lambda_1 \cdot \text{sign}(\Delta \text{goals})**r**terminal**=**λ**1**⋅**sign**(**Δ**goals**)**
* \lambda_2 \cdot \Delta \text{goals}

  ]

  with λ1\lambda_1**λ**1 dominating and λ2\lambda_2**λ**2 small (e.g., λ1=1,λ2=0.1\lambda_1=1,\lambda_2=0.1**λ**1**=**1**,**λ**2**=**0.1**). This is standard in football RL and strongly pushes the agent to increase goal difference even after securing a lead.**offline-rl-neurips.github**+1****
* **Dense offensive shaping (per time step):**

  * Small reward for  **puck x‑velocity toward opponent goal** .
  * Small reward for  **puck inside opponent half** .
  * Small reward for  **shots on goal or hitting goal posts** .
  * Possibly a *tiny* negative reward for being too deep in your own half when leading (to discourage “parking the bus”).

These are analogous to the “checkpoint” and positional density rewards used in football RL, where they kept these **at most O(0.1) of the terminal goal reward** to avoid overriding the main objective.[fse.studenttheses.rug](https://fse.studenttheses.ub.rug.nl/33947/1/bAI2024vanDommeleA.pdf)

## Constant vs adaptive tie penalties

Literature:

* **Constant tie values** (e.g., 0.5, 0, 0.2, −2) are used in many board game RL examples to encode fixed preferences.**youtube****kth.diva-portal**+2
* **Adaptive shaping** (dynamic PBRS, DRiP) is used to adjust **shaping terms** over time or state, but  **terminal outcome values are usually kept fixed** .**ifaamas**+1

For your use case:

* Use a **fixed terminal tie value** (e.g., −0.4), and
* Combine it with **adaptive *per‑step* shaping** (time‑dependent stalemate penalties, offensive bonuses) as above.

This is the safest combination:  **terminal reward encodes what you truly care about** , shaping just makes it learnable.**lilianweng.github**+2

## 1.3. Impact and caveats

* **Impact (priority):** **High.** Draw reward and shaping directly control your “aggressive vs defensive” balance and are a likely root cause of the 44% ties.
* **Trade‑offs:**
  * Over‑penalizing draws (e.g., tie as bad as or worse than loss) can create  **wildly over‑aggressive, unstable policies** , especially in self‑play, and may hurt TrueSkill if the agent throws safe wins away.**lilianweng.github**+1
  * Dense shaping risks **reward hacking** (agent optimizes “shots” without caring about goals, etc.); PBRS‑style shaping and careful scaling can mitigate this.**lilianweng.github**+1
* **Hockey‑specific:** Existing winning agents for laser‑hockey/hockey‑env achieve low draw rates *without* explicit draw penalties, via  **dense positional shaping + strong policies + diverse opponents** . You are already strong, so **a moderate tie penalty plus offensive shaping** is a natural next step.**github**+2

---

## CHALLENGE 2 – Rapidly Switching Opponents and Unstable Targets

## 2.1. How large-scale systems schedule opponents

**AlphaStar (StarCraft II):**

* Uses a **league** with:
  * Main agents,
  * Main exploiters,
  * League exploiters, plus many historical snapshots.**nature**+1
* **Opponent sampling** uses  **Prioritized Fictitious Self‑Play (PFSP)** :
  * For main agents, the opponent is drawn from a candidate set C\mathcal{C}**C** with

    Pr⁡(B∣A)∝f(p(A beats B))\Pr(B\mid A) \propto f(p(A \text{ beats } B))**Pr**(**B**∣**A**)**∝**f**(**p**(**A** beats **B**))**
    where:

    * fhard(x)=(1−x)pf_{\text{hard}}(x) = (1-x)^p**f**hard**(**x**)**=**(**1**−**x**)**p focuses on **hardest opponents** (winrate close to 0),
    * fvar(x)=x(1−x)f_{\text{var}}(x) = x(1-x)**f**var**(**x**)**=**x**(**1**−**x**) focuses on **similar‑strength opponents** (winrate near 0.5).**proceedings.neurips**+2****
  * Ablations show **PFSP + some self‑play** (about a 50–50 mix in their simplified ablation) gives the best Elo and least forgetting; pure self‑play is strong but more prone to cycles and forgetting.[googleapis](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
* Opponents are  **sampled per game** , but because the league is large and training is massively parallel, each parameter update sees a  **relatively stationary mixture** .**arxiv**+2

**OpenAI Five (Dota 2):**

* Uses PPO with massive self‑play.**openai**+1
* **Opponent scheduling** :
* **80% of training games** are played against the  **latest policy** , and  **20% vs past versions** .**himanshusahni.github**+1
* Past opponents are sampled using a **softmax over a “quality” score qiq_i**q**i** stored per opponent; after beating an opponent, its qiq_i**q**i is decreased to reduce its future sampling probability.[openai](https://cdn.openai.com/dota-2.pdf)
* This is effectively a **PFSP‑like mechanism** over past versions: more attention to  **stronger, harder opponents** , while not forgetting earlier ones.

**General self‑play & curriculum frameworks:**

* The **Syllabus** suite and associated self-play survey categorize self-play algorithms by how they maintain an opponent population and sample from it (SP, FSP, PFSP, PSRO, etc.).**ijcai**+4
* **PFSP** is widely used as the “default” in competitive games: maintain a  *history of past opponents* , sample according to some function of winrate (hardest or equal‑strength first).**arxiv**+2
* Unity ML‑Agents competitive self‑play examples:
  * Snapshot the main policy every  **`swap_steps=10000` steps** , and
  * Sample opponent with 50% chance from the **latest model** and 50% from earlier checkpoints.[gocoder](https://www.gocoder.one/blog/competitive-self-play-unity-ml-agents/)
* Da Silva et al.’s **SEPLEM** builds a self‑play curriculum by  **occasional play vs an expert** , modeling that expert, and then training in synthetic environments; this supports the idea of **distinct phases or blocks** of training per opponent.[utexas](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/BRACIS2019-Leno.pdf)

## 2.2. Frequency of opponent switching and batching

There is **no explicit theorem** about “correct number of episodes per opponent”, but the patterns above actually suggest a compromise for your situation:

* **Switching every episode with a small pool** (weak, strong, self‑play) in an off‑policy TD3 setting means:
  * Your replay buffer contains  **mixed, highly non‑stationary data** .
  * The Q‑function is trying to approximate values vs a distribution that changes faster than the TD updates can track.**reddit**+1
* In large systems, **policy updates lag opponent updates** by thousands of games, so each update sees a quasi‑stationary distribution.**googleapis**+1

For your scale (single TD3, 100k episodes):

* Use  **episode blocks** :
  * Sample an opponent type (weak, strong, self‑play) according to your mixture (e.g., 70% anchor, 30% pool).
  * Fix that opponent for  **K episodes** , where  **K≈10–30** , before sampling a new opponent.
* Reasoning:
  * K=10 episodes × 250 steps ≈ 2500 transitions, enough to give the Q‑function a coherent “picture” of that opponent.
  * K=30–50 may be even better early, then reduce later once the policy is more stable.

This is similar in spirit to **“swap_steps” in ML‑Agents** (10k environment steps between opponent swaps) and to **league “iterations”** where new opponents are added between training phases, not at every game.**arxiv**+3

## 2.3. Anchor vs self‑play mixture

Large systems:

* AlphaStar’s main agents see a mixture of  **self‑play (SP) and PFSP vs the league** ; the ablation uses a  **50–50 mix** , but the full system also includes exploiters and other roles.[googleapis](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
* OpenAI Five uses  **80% latest vs 20% historical policies** , which function as “anchors” for older behaviors.**himanshusahni.github**+1

Your current **70% anchor / 30% self‑play pool** is in the same ballpark, but you could refine it:

* Early training:
  * Stick to roughly **70% basic/weak/strong scripted opponents** and  **30% self‑play** , to stabilize TD3’s off‑policy updates.
* Later training (after ~30–50k episodes):
  * Move toward  **50–60% anchor, 40–50% self‑play** , especially once you’re consistently beating the basic opponents by a wide margin.

Crucially, make the  **mixture stable over blocks** , not per episode: e.g., maintain a running schedule like:

* Sample opponent type distribution  *(weak 40%, strong 30%, self‑play pool 30%)* .
* For each draw, play **K episodes in a row vs that type** before resampling.

## 2.4. Dual replay buffers and sampling strategies

Research on  **multi-buffer or dual-buffer replay** :

* **Double replay memory** for DQN: keep **two replay memories** – one focusing on **important transitions** (e.g., high TD‑error or rare events) and one on  **new transitions** , and sample from both; improves performance over a single buffer.[onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1155/2021/6652042)
* **Dual experience replay buffer** in asynchronous DRL (TPDEB): combines a **uniform buffer** with a  **prioritized trajectory buffer** , sampling from both to balance unbiased sampling and learning from valuable trajectories.[journals.plos](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0334411)
* **Dual-memory integrated learning** in MARL: combines short‑term and long‑term memories in a dual architecture to improve sample efficiency and keep learning stable.[arxiv](https://arxiv.org/html/2407.16034v1)
* **Adaptive replay buffer training** : suggests there is an  *optimal buffer size and sampling regime* , and deviating from it (too small or too large) hurts convergence.[emergentmind](https://www.emergentmind.com/topics/adaptive-replay-buffer-training)
* Reddit discussions on off‑policy self‑play emphasize that with a single buffer,  **old experiences vs obsolete opponents can dominate** , and recommend  **smaller buffers or biased sampling toward recent data** .[reddit](https://www.reddit.com/r/reinforcementlearning/comments/qn6x72/can_i_use_selfplay_in_an_offpolicy_setting/)

For your setup (anchor vs pool opponents + self‑play), two practical designs:

1. **Separate buffers per opponent class:**
   * **Buffer A:** experiences vs anchor/basic opponents.
   * **Buffer B:** experiences vs self‑play / pool opponents.
   * For each TD3 update, sample **a fixed proportion** from each buffer:
     * Early: e.g.,  **70% batch from A, 30% from B** .
     * Later: transition to **50/50** or even 40/60.
   * This guarantees the critic always sees enough anchor data to stay grounded, while still learning to exploit the evolving self‑play pool.
2. **Single buffer with opponent tags and stratified sampling:**
   * Keep **one replay buffer** but add an **opponent_type** field in each transition (weak/strong/self‑play + possibly specific opponent ID).
   * When sampling, use a  **stratified sampler** : e.g., ensure each mini‑batch contains at least 30% transitions from basic opponents and 70% from pool/self‑play, or maintain a balanced distribution by sampling subsets by tag.

Both are in line with the **dual‑buffer / dual‑memory ideas** above, but tailored to opponent type instead of TD error.**arxiv**+2

## 2.5. PSRO / PFSP and code examples

* **PSRO (Policy-Space Response Oracles)** and its variants (anytime PSRO, P2SRO) generalize double oracle and FSP, iteratively building a **population of policies** and computing responses to a  **mixture over that population** .**cmu**+2
* **PFSP** is implemented in several open frameworks:
  * The **Syllabus** library includes FSP and PFSP as built‑in curricula (maintain opponent history, select hardest or equal‑strength opponent via winrate).[arxiv](https://arxiv.org/html/2411.11318v2)
  * The **ROA‑Star** improvement of AlphaStar explicitly uses PFSP to choose “hardest opponents” from the league and shows worst‑case winrate improvements over baseline AlphaStar in StarCraft II.[proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf)
* **OpenAI‑style PFSP** (softmax over quality) is described in detail in their Dota 2 paper: each past opponent has a **quality score qiq_i**q**i** and is sampled with probability proportional to exp⁡(qi)\exp(q_i)**exp**(**q**i**)**, updated based on how often the current policy beats it.[openai](https://cdn.openai.com/dota-2.pdf)

For coding in your TD3 loop, a simple PFSP variant over your pool of checkpoints:

* Maintain for each opponent ii**i** a **winrate estimate p^i\hat{p}_i**p**^**i**** against the current policy, updated from evaluation games.
* Define sampling probability:
  * **Equal-strength PFSP:** wi=p^i(1−p^i)w_i = \hat{p}_i (1 - \hat{p}_i)**w**i**=**p**^**i**(**1**−**p**^**i**)**.
  * **Hardest‑first PFSP:** wi=(1−p^i)pw_i = (1 - \hat{p}_i)^p**w**i**=**(**1**−**p**^**i**)**p** with p∈[1,3]p \in [1,3]**p**∈**[**1**,**3**]**.
* Normalize to get a distribution over pool opponents; sample an opponent, then play K episodes vs it.

## 2.6. Impact and caveats

* **Impact (priority):** **High.** Against non‑stationary opponents, off‑policy TD3 is very sensitive to the **stationarity of the replay data** and the  **relative frequency of each opponent type** .
* **Trade‑offs:**
  * Longer blocks per opponent reduce non‑stationarity but **may cause local overfitting** if K is too large.
  * Too much anchor opponent play can lead to  **over‑specialization** ; too little anchor play makes training unstable.
  * Dual‑buffer mechanisms increase complexity and memory use, but the literature suggests  **clear benefits in stability and sample efficiency** .**emergentmind**+3
* **Hockey‑specific:** Winning laser‑hockey projects at Tübingen used **basic opponent + checkpoint pool** and  **sampling from historical checkpoints** ; your architecture is conceptually similar, but you are running much longer and with an off‑policy method, so stabilizing opponent scheduling becomes more important than in the original coursework setups.**github**+3

---

## CHALLENGE 3 – Hyperparameter Scaling for 100k‑Episode TD3 Training

Here 100k episodes × ~250 steps ≈  **25M transitions** .

## 3.1. Learning rate scheduling in long‑running RL

Evidence:

* **SEERL:** Uses **cosine cyclical annealing** of LR over 1M timesteps, with multiple cycles; finds that LR annealing yields **better generalization** than a small constant LR, and that **higher initial LR** shapes which local minima are found.[ifaamas](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1100.pdf)
* **FastTD3:** Trains TD3 with **large batch sizes and tuned hyperparameters** to solve humanoid tasks in under 3 hours, showing that TD3 remains stable with  **larger batches and appropriately chosen LR** , and benefiting from learning‑rate tuning.[arxiv](https://arxiv.org/html/2505.22642v1)
* **OpenAI Five:** Runs PPO with:
  * Very large batches (~1–3M timesteps),
  * Careful control of **data staleness** and  **sample reuse** ,
  * LR tuned for long‑horizon training; though they do not publish a fancy schedule, they show that  **learning is very sensitive to staleness and reuse** .[openai](https://cdn.openai.com/dota-2.pdf)
* Generic RL LR guides confirm that **decaying LR over long runs** (step‑wise or cosine) tends to improve final performance and stability compared to a high constant LR.**reinforcementlearningpath**+1

For 25M steps, a **simple but effective schedule** is:

* Let  **TT**T** = total gradient update steps** .
* Use a **cosine decay** from η0\eta_0**η**0 to ηmin⁡\eta_{\min}**η**m**i**n:

  η(t)=ηmin⁡+12(η0−ηmin⁡)[1+cos⁡(πtT)]\eta(t) = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})\left[1 + \cos\left(\pi \tfrac{t}{T}\right)\right]**η**(**t**)**=**η**m**i**n**+**2**1**(**η**0**−**η**m**i**n**)**[**1**+**cos**(**π**T**t**)**]**
* For TD3:

  * Critic LR: η0,c≈3⋅10−4\eta_{0,c} \approx 3\cdot 10^{-4}**η**0**,**c**≈**3**⋅**1**0**−**4**, ηmin⁡,c≈3⋅10−5\eta_{\min,c} \approx 3\cdot 10^{-5}**η**m**i**n**,**c**≈**3**⋅**1**0**−**5**.
  * Actor LR: η0,a≈3⋅10−4\eta_{0,a} \approx 3\cdot 10^{-4}**η**0**,**a**≈**3**⋅**1**0**−**4**, ηmin⁡,a≈3⋅10−5\eta_{\min,a} \approx 3\cdot 10^{-5}**η**m**i**n**,**a**≈**3**⋅**1**0**−**5**.

Alternatively, a  **2–3 step schedule** :

* 0–30% training: LR = η0\eta_0**η**0.
* 30–70%: LR = η0/3\eta_0 / 3**η**0**/3**.
* 70–100%: LR = η0/10\eta_0 / 10**η**0**/10**.

This matches common continuous‑control baselines and the general findings that LR annealing improves robustness for long training.**reinforcementlearningpath**+2

## 3.2. Exploration scheduling

In TD3, exploration is typically produced by  **action noise** , not an ε\varepsilon**ε** over random actions. Your description (“ε\varepsilon**ε** decays from 0.999 to 0.34 at 45k episodes, reaches 0.1 at ~70k”) suggests an  **additional exploration mechanism** .

Relevant signals:

* In standard continuous control, exploration is often  **high early and lower late** ; decaying noise standard deviation over training is common practice.
* In long-horizon tasks, too little exploration early can prevent convergence; too much persistent exploration late can **destabilize** convergence.[arxiv](https://arxiv.org/html/2403.09583v4)
* ExploRLLM, on long-horizon tasks, finds that **exploration frequencies ϵ∈(0,0.5]\epsilon \in (0, 0.5]**ϵ**∈**(**0**,**0.5**]**** (in their sense of “LLM-based exploration moves”) produce  **fastest convergence** , whereas **too high exploration (>0.5)** slows or prevents convergence.[arxiv](https://arxiv.org/html/2403.09583v4)

For your 100k‑episode run:

* **Action noise schedule (TD3‑style):**
  * Start with **Gaussian noise σ ≈ 0.2–0.3** relative to action range.
  * Linearly decay σ to ~0.05 over the first  **50–70% of training** , then keep it constant.
* If you also use an  **ε\varepsilon**ε**-greedy override** :
  * Start at ε0≈0.8\varepsilon_0 \approx 0.8**ε**0**≈**0.8, decay to  **0.1–0.2 by about 50k episodes** , and perhaps to  **0.05 by 80k** .
  * Avoid ε\varepsilon**ε** near 1 for too long; with a good warmup phase and large replay, pure random actions deep into training mostly degrade data quality.

Given 25M steps, **front‑load more exploration** (higher noise and higher ε\varepsilon**ε**) in the **first 10–20% of steps** rather than spreading it too evenly over the entire run.

## 3.3. Replay buffer sizing

Evidence:

* Classic DQN on Atari used a **1M‑transition replay buffer** for 200M frames (0.5% retention), and this remains a standard baseline.
* **Pruning replay buffers** (e.g., TRBP) shows that **pruning up to ~50% of the buffer** can reduce memory without major performance loss, but excessively small buffers do hurt performance.[emerginginvestigators](https://emerginginvestigators.org/articles/23-068/pdf)
* **Adaptive replay buffer training** shows that there is an  **optimal buffer size** : too small leads to overshooting (overfitting latest data), too large oversmooths and slows learning; adaptive methods change memory size based on TD error and learning progress.[emergentmind](https://www.emergentmind.com/topics/adaptive-replay-buffer-training)
* OpenAI Five experiments show that:
  * **Sample reuse ≈ 1** (each sample used once) works best,
  * Excess reuse (>2–3×)  **slows learning dramatically** .[openai](https://cdn.openai.com/dota-2.pdf)

You have  **25M total transitions** ; a **500k buffer** stores only 2%. That is not necessarily too small – Atari RL often keeps ~0.5% – but in a **non‑stationary opponent setting** the trade‑off is different:

* Too large a buffer keeps a lot of obsolete opponent data.
* Too small makes the Q‑function overfit recent opponents and forget older ones.

A reasonable compromise:

* Increase buffer size to **1–2M transitions** (4–8% of total).
* Keep **sample reuse close to 1** by:
  * Ensuring that you do not perform many gradient steps per environment step,
  * Or by limiting how often each sample is drawn (if you implement more updates per step).

If GPU memory is tight, start with **1M** and monitor learning curves; use buffer tagging (by opponent type and “age”) and a **biased replacement scheme** (e.g., dropping oldest data from obsolete opponents first) to keep the effective buffer “fresh” regarding current opponent distribution.**reddit**+1

## 3.4. Batch size for extended training

Evidence:

* OpenAI Five: scaling batch size from ~123k to ~983k timesteps improved training speed; **speedup was sublinear but significant** (≈2.5× faster at TS=175) and data quality (staleness, sample reuse) mattered more than batch size once too large.[openai](https://cdn.openai.com/dota-2.pdf)
* FastTD3: uses **very large batches (e.g., 8192)** with carefully tuned LR to achieve stable and fast convergence.[arxiv](https://arxiv.org/html/2505.22642v1)

For TD3 on a single GPU, **batch size 512** is already substantial:

* It is large enough that gradient estimates are low‑variance.
* Increasing to **1024** can improve stability slightly if your GPU allows it, but the main gains will come from:
  * Proper LR scheduling,
  * Better replay composition, and
  * Opponent scheduling, not from further batch size increases.

There is **no strong evidence** that **increasing batch size over time** (e.g., growing batch as training progresses) is beneficial in RL; better to **keep it fixed** and adjust LR and replay behavior.**ifaamas**+2

## 3.5. Warmup phase scaling

In off‑policy continuous control (DDPG, TD3):

* Typical implementations use a  **random policy for the first N steps** :
  * TD3 original paper uses ~25k random steps in a 1M‑step run (2.5% of steps).[arxiv](https://arxiv.org/pdf/1802.09477.pdf)
* This ensures the replay buffer is reasonably diverse before bootstrapping with Q‑learning.

You currently use  **500 warmup episodes** :

* 500 × 250 ≈ 125k random steps =  **0.5% of 25M** , which is **shorter than typical** if measured as a fraction of total steps.

Given your long training:

* Increase warmup to  **2–5% of total steps** :
  * That’s **500k–1.25M random steps** (≈2000–5000 episodes).
* Or use a  **gradual warmup** :
  * First X steps: pure random actions.
  * Next Y steps: large action noise and high ε\varepsilon**ε**.
  * Then switch to your “normal” exploration schedule.

This aligns better with both TD3 original settings and long‑horizon continuous control practices.**arxiv**+1

## 3.6. Self-play pool size

In league/self‑play systems:

* AlphaStar’s league ended up with  **~900 distinct players** , but the **effective Nash mixture** placed most weight on  **recent players** , with small probabilities for older ones.[googleapis](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
* PSRO and related work emphasise that a **restricted but diverse population** (tens, not hundreds, of policies) is sufficient to approximate Nash equilibria well.**proceedings.neurips**+1
* Syllabus supports  **FSP and PFSP over past opponents** , assuming a  **history of many opponents** , but practical curricula often operate on a  **dozen to a few dozen policies** .[arxiv](https://arxiv.org/html/2411.11318v2)

Your current  **12 checkpoint opponents** :

* For a 100k‑episode training run, this is probably  **on the small side** :
  * It under‑represents the diversity of strategies your agent encounters over such a long training period.

Recommended:

* Increase pool max to about  **20–30 checkpoints** .
* Use **PFSP** to sample among them based on winrate.
* Consider **“retirement”** of clearly dominated or identical opponents (e.g., those with extremely low or high winrate and low diversity).

There is little evidence that going beyond ~50 adds much; the main gains are between **a handful** of opponents and  **a few dozen** .**ijcai**+3

## 3.7. Evaluation frequency (with TrueSkill)

In long‑running self‑play systems:

* OpenAI Five:
  * Used **TrueSkill** over a pool of ~83 reference agents,
  * Updated TrueSkill stats every  **couple of hours** , with hundreds of evaluation games, and showed that **a TrueSkill difference of ≈8.3 corresponds to an 80% winrate** between two agents.**arxiv**+1
* Recent LLM tournament work also uses TrueSkill; they find that **tournaments with ~100–200 matches per agent** are sufficient to get stable rankings, and that TrueSkill converges faster than Elo in multi‑player settings.**arxiv**+3
* Ghost Recon matchmaking analysis shows that using only win/loss TrueSkill (“TrueSkill‑Team”) converges slower than  **TrueSkill variants augmented with in‑game performance scores** , but still works well when enough games are played.[iro.umontreal](https://www.iro.umontreal.ca/~lisa/pointeurs/gro-matchmaking-ieee.pdf)

For your 100k‑episode run:

* **Per‑episode evaluation** (e.g., every 100 episodes) is likely too noisy and expensive.
* Recommended evaluation schedule:
  * **Early phase (0–20k episodes):**
    * Evaluate every  **500 episodes** .
    * Each evaluation: play **50–100 games** against:
      * Weak basic opponent,
      * Strong basic opponent,
      * A  **small subset of pool opponents** ,
      * Optionally a  **frozen historical “reference” agent** .
  * **Mid/late phase (20k–100k episodes):**
    * Evaluate every  **2000–5000 episodes** .
    * Increase evaluation breadth (more pool opponents, more games) as you approach final training.
* Maintain a **TrueSkill ladder** over a  **fixed set of reference agents** :
  * A handful of weak/strong/basic scripted opponents,
  * Several frozen checkpoints of your own agent,
  * Optionally a few “challenger” agents from different seeds.
  * Use standard TrueSkill parameters (e.g. μ0=25,σ0=25/3\mu_0 = 25, \sigma_0 = 25/3**μ**0**=**25**,**σ**0**=**25/3**) as in other tournament evaluations.**microsoft**+5****

This gives you a  **stationary tournament evaluation** , decoupled from the constantly evolving training pool.

## 3.8. Impact and caveats

* **Impact (priority):** **Medium–High.** Your current hyperparameters aren’t obviously wrong, but scaling from 10k–50k to 100k episodes without adjusting LR, buffer, warmup, and evaluation will definitely  **expose weaknesses** .
* **Trade‑offs:**
  * Larger replay and longer warmup increase **wall‑clock time** before seeing strong performance.
  * LR decay can slow adaptation to late‑training changes (e.g., new opponents), so you might want **slow decay** or **cosine with a high floor** rather than decaying to almost zero.
  * Larger self‑play pool and more evaluation games increase  **compute and memory** , but the benefit to robustness is high for a tournament scenario.

---

## Bonus: TrueSkill-based Tournament Evaluation and Batch vs Step Training

**TrueSkill and tournaments:**

* Various works (LLM tournaments, SKATE, TextArena) show that **TrueSkill is well suited to repeated tournaments** with many players; it handles **multi-player games, variable numbers of participants, and draws** well.**arxiv**+3
* **TrueSkill 2** and industry case studies (e.g., Ghost Recon Online) show that including **in‑game performance features** (e.g., score, kills) in ranking can speed convergence, but vanilla TrueSkill with win/draw/loss also works, given enough matches.**microsoft**+1
* OpenAI Five’s **TrueSkill chart** over training is a good template: TrueSkill difference ≈8.3 ≈ 80% winrate gives you an interpretable scale for your hockey agents.[openai](https://cdn.openai.com/dota-2.pdf)

For your tournament:

* Use a **TrueSkill ladder** with:
  * Fixed initial priors,
  * Stable reference opponents,
  * Enough matches per pairing to reduce uncertainty.

**Batch vs episode-wise training:**

* A Reddit discussion comparing **step‑wise vs episode‑wise training** in MountainCar found that:
  * Step‑wise updates (small batch after each step) gave **better convergence** than one large update per episode, even when total samples were equal.[reddit](https://www.reddit.com/r/reinforcementlearning/comments/18qgg22/rl_training_in_episodes_instead_of_steps/)
  * Episode‑wise updates led to poor convergence and difficulty “solidifying” the learned behavior.
* OpenAI Five’s experiments on **batch size and data staleness** suggest that:
  * Larger batches help, but **using fresh data** (low staleness, low reuse) is more important.[openai](https://cdn.openai.com/dota-2.pdf)
  * Delaying updates until entire episodes are finished (and parameters have moved many steps) hurts learning.

For your TD3 agent:

* Continue using  **step‑based updates with off‑policy replay** , rather than pure episode‑by‑episode updates.
* Ensure that **data staleness** between collection and update is small (in your setting this is naturally true).

---

## Priorities to Fix in Practice

1. **Reward & tie handling (High):**
   * Introduce **tie penalty** in the range **−0.3 to −0.5** (scaled to your win/loss values).
   * Add **time‑dependent stalemate penalties** and  **goal‑differential/offensive shaping** .
   * Monitor changes in tie rate and TrueSkill vs weak/strong opponents.
2. **Opponent & replay scheduling (High):**
   * Introduce **PFSP‑style opponent sampling** for your self‑play pool.
   * Play in  **blocks of 10–30 episodes per opponent** .
   * Use **dual buffers or stratified sampling** over anchor vs pool transitions.
3. **Long‑run hyperparameters (Medium–High):**
   * Implement **LR decay** over 25M steps (cosine or step schedule).
   * Increase **warmup** to ~2–5% of total steps.
   * Grow **replay buffer** to ~1–2M transitions and keep  **sample reuse ≈1** .
   * Expand **self‑play pool** to ~20–30 opponents.
   * Adjust **evaluation frequency** (500 episodes early, 2–5k later) and move to **TrueSkill‑based ranking** for tournament readiness.

With these changes, your TD3 self‑play system will be much closer in spirit to the designs of  **AlphaStar** ,  **OpenAI Five** , and modern multi‑agent curricula, while being tailored to the specific quirks of hockey‑env and a 100k‑episode budget.

1. [https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
2. [https://www.youtube.com/watch?v=1m33PKLs5Z8](https://www.youtube.com/watch?v=1m33PKLs5Z8)
3. [https://github.com/antirez/ttt-rl](https://github.com/antirez/ttt-rl)
4. [https://kth.diva-portal.org/smash/get/diva2:1795583/FULLTEXT01.pdf](https://kth.diva-portal.org/smash/get/diva2:1795583/FULLTEXT01.pdf)
5. [https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
6. [https://cdn.openai.com/dota-2.pdf](https://cdn.openai.com/dota-2.pdf)
7. [https://himanshusahni.github.io/2018-09-18-open_ai_five_and_the_limits_of_self_play.html](https://himanshusahni.github.io/2018-09-18-open_ai_five_and_the_limits_of_self_play.html)
8. [https://proceedings.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf)
9. [https://www.gocoder.one/blog/competitive-self-play-unity-ml-agents/](https://www.gocoder.one/blog/competitive-self-play-unity-ml-agents/)
10. [https://www.reddit.com/r/reinforcementlearning/comments/qn6x72/can_i_use_selfplay_in_an_offpolicy_setting/](https://www.reddit.com/r/reinforcementlearning/comments/qn6x72/can_i_use_selfplay_in_an_offpolicy_setting/)
11. [https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1100.pdf](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1100.pdf)
12. [https://arxiv.org/html/2505.22642v1](https://arxiv.org/html/2505.22642v1)
13. [https://arxiv.org/html/2411.11318v2](https://arxiv.org/html/2411.11318v2)
14. [https://arxiv.org/html/2408.01072v1](https://arxiv.org/html/2408.01072v1)
15. [https://openreview.net/pdf/6d523bdc8240fc5345ad1d2aa36fe3367b5263ae.pdf](https://openreview.net/pdf/6d523bdc8240fc5345ad1d2aa36fe3367b5263ae.pdf)
16. [https://arxiv.org/html/2508.06111v1](https://arxiv.org/html/2508.06111v1)
17. [https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/trueskill2.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/trueskill2.pdf)
18. [https://arxiv.org/html/2504.11442v1](https://arxiv.org/html/2504.11442v1)
19. [https://github.com/anticdimi/laser-hockey](https://github.com/anticdimi/laser-hockey)
20. [https://github.com/meier-johannes94/Reinforcement-Learning-PPO](https://github.com/meier-johannes94/Reinforcement-Learning-PPO)
21. [https://github.com/the-klingspor/laser-hockey](https://github.com/the-klingspor/laser-hockey)
22. [https://github.com/kitteltom/rl-laser-hockey](https://github.com/kitteltom/rl-laser-hockey)
23. [https://www.ias.informatik.tu-darmstadt.de/uploads/Team/PuzeLiu/AirHockeyChallenge_SpaceR.pdf](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/PuzeLiu/AirHockeyChallenge_SpaceR.pdf)
24. [https://github.com/AndrejOrsula/drl_air_hockey](https://github.com/AndrejOrsula/drl_air_hockey)
25. [https://fse.studenttheses.ub.rug.nl/33947/1/bAI2024vanDommeleA.pdf](https://fse.studenttheses.ub.rug.nl/33947/1/bAI2024vanDommeleA.pdf)
26. [https://offline-rl-neurips.github.io/2021/pdf/6.pdf](https://offline-rl-neurips.github.io/2021/pdf/6.pdf)
27. [https://arxiv.org/html/2408.10215v1](https://arxiv.org/html/2408.10215v1)
28. [https://www.ifaamas.org/Proceedings/aamas2014/aamas/p165.pdf](https://www.ifaamas.org/Proceedings/aamas2014/aamas/p165.pdf)
29. [https://sites.google.com/view/fightladder/home](https://sites.google.com/view/fightladder/home)
30. [https://lilianweng.github.io/posts/2024-11-28-reward-hacking/](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
31. [https://www.nature.com/articles/s41586-019-1724-z](https://www.nature.com/articles/s41586-019-1724-z)
32. [https://arxiv.org/html/2408.01072v3](https://arxiv.org/html/2408.01072v3)
33. [https://arxiv.org/abs/1912.06680](https://arxiv.org/abs/1912.06680)
34. [https://www.ijcai.org/proceedings/2024/0880.pdf](https://www.ijcai.org/proceedings/2024/0880.pdf)
35. [https://proceedings.neurips.cc/paper_files/paper/2023/file/d61819e9b4a607b8448de762235148c4-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/d61819e9b4a607b8448de762235148c4-Paper-Conference.pdf)
36. [https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/BRACIS2019-Leno.pdf](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/BRACIS2019-Leno.pdf)
37. [https://onlinelibrary.wiley.com/doi/10.1155/2021/6652042](https://onlinelibrary.wiley.com/doi/10.1155/2021/6652042)
38. [https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0334411](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0334411)
39. [https://arxiv.org/html/2407.16034v1](https://arxiv.org/html/2407.16034v1)
40. [https://www.emergentmind.com/topics/adaptive-replay-buffer-training](https://www.emergentmind.com/topics/adaptive-replay-buffer-training)
41. [https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture21-Slides.pdf](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture21-Slides.pdf)
42. [https://www.reinforcementlearningpath.com/the-complete-guide-of-learning-rate-in-rl](https://www.reinforcementlearningpath.com/the-complete-guide-of-learning-rate-in-rl)
43. [https://arxiv.org/html/2403.09583v4](https://arxiv.org/html/2403.09583v4)
44. [https://emerginginvestigators.org/articles/23-068/pdf](https://emerginginvestigators.org/articles/23-068/pdf)
45. [https://arxiv.org/pdf/1802.09477.pdf](https://arxiv.org/pdf/1802.09477.pdf)
46. [https://www.iro.umontreal.ca/~lisa/pointeurs/gro-matchmaking-ieee.pdf](https://www.iro.umontreal.ca/~lisa/pointeurs/gro-matchmaking-ieee.pdf)
47. [https://www.ijcai.org/Proceedings/15/Papers/049.pdf](https://www.ijcai.org/Proceedings/15/Papers/049.pdf)
48. [https://www.reddit.com/r/reinforcementlearning/comments/18qgg22/rl_training_in_episodes_instead_of_steps/](https://www.reddit.com/r/reinforcementlearning/comments/18qgg22/rl_training_in_episodes_instead_of_steps/)
49. [https://github.com/martius-lab/hockey-env](https://github.com/martius-lab/hockey-env)
50. [https://github.com/martius-lab](https://github.com/martius-lab)
51. [https://github.com/naivoder/TD3](https://github.com/naivoder/TD3)
52. [https://github.com/triooy/rl_uni_tuebingen_challenge](https://github.com/triooy/rl_uni_tuebingen_challenge)
53. [https://www.mathworks.com/help/reinforcement-learning/ug/td3-agents.html](https://www.mathworks.com/help/reinforcement-learning/ug/td3-agents.html)
54. [https://joeydotcomputer.substack.com/p/trueskill-2-online-matchmaking-meta](https://joeydotcomputer.substack.com/p/trueskill-2-online-matchmaking-meta)
55. [https://www.youtube.com/watch?v=ZhFO8EWADmY](https://www.youtube.com/watch?v=ZhFO8EWADmY)
56. [https://danieltakeshi.github.io/2020/06/28/offline-rl/](https://danieltakeshi.github.io/2020/06/28/offline-rl/)
57. [https://intellabs.github.io/coach/components/agents/policy_optimization/td3.html](https://intellabs.github.io/coach/components/agents/policy_optimization/td3.html)
58. [http://papers.neurips.cc/paper/8484-sample-efficient-deep-reinforcement-learning-via-episodic-backward-update.pdf](http://papers.neurips.cc/paper/8484-sample-efficient-deep-reinforcement-learning-via-episodic-backward-update.pdf)
59. [https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_136.pdf](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_136.pdf)
60. [https://docs.pytorch.org/rl/0.4/tutorials/rb_tutorial.html](https://docs.pytorch.org/rl/0.4/tutorials/rb_tutorial.html)
61. [https://liu.diva-portal.org/smash/get/diva2:1687088/FULLTEXT01.pdf](https://liu.diva-portal.org/smash/get/diva2:1687088/FULLTEXT01.pdf)
62. [https://www.jmlr.org/papers/volume26/24-1503/24-1503.pdf](https://www.jmlr.org/papers/volume26/24-1503/24-1503.pdf)
63. [https://github.com/davidADSP/SIMPLE](https://github.com/davidADSP/SIMPLE)
64. [https://academic.oup.com/jcde/article/10/2/830/7069331](https://academic.oup.com/jcde/article/10/2/830/7069331)
65. [https://deepmind.google/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/](https://deepmind.google/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/)
66. [https://www.academia.edu/97146962/Grandmaster_level_in_StarCraft_II_using_multi_agent_reinforcement_learning](https://www.academia.edu/97146962/Grandmaster_level_in_StarCraft_II_using_multi_agent_reinforcement_learning)
67. [https://arxiv.org/abs/1911.08265](https://arxiv.org/abs/1911.08265)
68. [https://kr2ml.github.io/2019/papers/KR2ML_2019_paper_57.pdf](https://kr2ml.github.io/2019/papers/KR2ML_2019_paper_57.pdf)
69. [https://www.emergentmind.com/topics/twin-delayed-deep-deterministic-policy-gradient-td3](https://www.emergentmind.com/topics/twin-delayed-deep-deterministic-policy-gradient-td3)
70. [https://arxiv.org/pdf/2102.13012.pdf](https://arxiv.org/pdf/2102.13012.pdf)
71. [https://bnaic2024.sites.uu.nl/wp-content/uploads/sites/986/2024/10/Reinforcement-Learning-of-Action-Sequences-in-Table-Football.pdf](https://bnaic2024.sites.uu.nl/wp-content/uploads/sites/986/2024/10/Reinforcement-Learning-of-Action-Sequences-in-Table-Football.pdf)
72. [https://universaar.uni-saarland.de/bitstream/20.500.11880/39965/1/thesis_rati_devidze.pdf](https://universaar.uni-saarland.de/bitstream/20.500.11880/39965/1/thesis_rati_devidze.pdf)
73. [https://proceedings.neurips.cc/paper_files/paper/2022/file/520425a5a4c2fb7f7fc345078b188201-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/520425a5a4c2fb7f7fc345078b188201-Paper-Conference.pdf)
74. [https://www.reddit.com/r/reinforcementlearning/comments/nzicoj/is_there_a_particular_reason_why_td3_is/](https://www.reddit.com/r/reinforcementlearning/comments/nzicoj/is_there_a_particular_reason_why_td3_is/)
75. [https://www.atlantis-press.com/proceedings/dai-23/125998066](https://www.atlantis-press.com/proceedings/dai-23/125998066)
76. [https://www.emergentmind.com/topics/self-play-training](https://www.emergentmind.com/topics/self-play-training)
77. [https://eitca.org/artificial-intelligence/eitc-ai-arl-advanced-reinforcement-learning/case-studies/aplhastar-mastering-startcraft-ii/examination-review-aplhastar-mastering-startcraft-ii/explain-the-self-play-approach-used-in-alphastars-reinforcement-learning-phase-how-did-playing-millions-of-games-against-its-own-versions-help-alphastar-refine-its-strategies/](https://eitca.org/artificial-intelligence/eitc-ai-arl-advanced-reinforcement-learning/case-studies/aplhastar-mastering-startcraft-ii/examination-review-aplhastar-mastering-startcraft-ii/explain-the-self-play-approach-used-in-alphastars-reinforcement-learning-phase-how-did-playing-millions-of-games-against-its-own-versions-help-alphastar-refine-its-strategies/)
78. [https://www.reddit.com/r/DotA2/comments/96tu37/understanding_openai_five_a_simplified_dissection/](https://www.reddit.com/r/DotA2/comments/96tu37/understanding_openai_five_a_simplified_dissection/)
79. [https://cs224r.stanford.edu/projects/pdfs/cs224r_final_report_2025%20(1)1.pdf](https://cs224r.stanford.edu/projects/pdfs/cs224r_final_report_2025%20(1)1.pdf)
80. [https://proceedings.mlr.press/v162/jeon22a/jeon22a.pdf](https://proceedings.mlr.press/v162/jeon22a/jeon22a.pdf)
81. [https://www.mathworks.com/help/reinforcement-learning/ref/rl.replay.rlreplaymemory.html](https://www.mathworks.com/help/reinforcement-learning/ref/rl.replay.rlreplaymemory.html)
82. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10400709/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10400709/)
83. [https://training.prodigalai.com/modules/rl-modules/chess-engine](https://training.prodigalai.com/modules/rl-modules/chess-engine)
84. [https://skywork.ai/skypage/en/Daily-AI-Research-Digest:-Google&#39;s-Reinforcement-Learning-Frontiers-for-Complex-Tasks/1948206470819074048](https://skywork.ai/skypage/en/Daily-AI-Research-Digest:-Google's-Reinforcement-Learning-Frontiers-for-Complex-Tasks/1948206470819074048)
85. [https://huggingface.co/learn/deep-rl-course/en/unit7/self-play](https://huggingface.co/learn/deep-rl-course/en/unit7/self-play)
86. [https://www.mathematik.uni-wuerzburg.de/fileadmin/10040900/2021/PlayingPongDQN.pdf](https://www.mathematik.uni-wuerzburg.de/fileadmin/10040900/2021/PlayingPongDQN.pdf)
87. [https://arxiv.org/html/2507.05465v1](https://arxiv.org/html/2507.05465v1)
88. [https://staff.fnwi.uva.nl/a.visser/education/bachelorAI/RubenVanHeusdenThesis.pdf](https://staff.fnwi.uva.nl/a.visser/education/bachelorAI/RubenVanHeusdenThesis.pdf)
89. [https://github.com/Ashish-Tripathy/TD3-Twin-Delayed-DDPG](https://github.com/Ashish-Tripathy/TD3-Twin-Delayed-DDPG)
90. [https://www3.hs-albsig.de/wordpress/point2pointmotion/2020/10/09/deep-reinforcement-learning-with-the-snake-game/](https://www3.hs-albsig.de/wordpress/point2pointmotion/2020/10/09/deep-reinforcement-learning-with-the-snake-game/)
91. [https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/paper-othello.pdf](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/paper-othello.pdf)
92. [https://www.reddit.com/r/learnprogramming/comments/132x5wn/in_reinforcement_learning_how_do_you_exactly/](https://www.reddit.com/r/learnprogramming/comments/132x5wn/in_reinforcement_learning_how_do_you_exactly/)
