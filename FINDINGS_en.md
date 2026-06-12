# Findings — contagion-fit (simple vs complex contagion fitted to real data)

Date: 2026-06-12
Context: ESADE Python assignment / project `contagion-fit`

> This document summarizes *what / based on what / how investigated / what was concluded*, down to the level of **equations and algorithms**.
> For a much gentler, picture-based version see [EXPLAINER_en.md](EXPLAINER_en.md). The Japanese counterparts are [FINDINGS_ja.md](FINDINGS_ja.md) / [EXPLAINER_ja.md](EXPLAINER_ja.md).

---

## 1. What we investigated (research question)

Which of the following two propagation mechanisms better reproduces the **size distribution** of real social-media information cascades?

- **Simple contagion (IC: Independent Cascade)** — each contact gives one stochastic trial with probability `p`.
- **Complex contagion (threshold model)** — a node activates only once the *fraction* of its already-active neighbours reaches a threshold `φ`.

And does the answer change with **content type** (science news / rumor), **platform** (Twitter / Weibo), and **substrate network structure**?

### Theoretical conflict in the background
- **Centola & Macy (2007)**: information spreads by simple contagion, behaviour by complex contagion. **Weak ties** accelerate simple contagion and impede complex contagion.
- **Watts & Dodds (2007)**: large cascades are driven not by influentials but by a critical mass of easily-influenced individuals.
- This project pits these debates against each other on the common ground of **fitting to real data**.

### How each claim is tested by this simulation

Each claim is translated into a concrete, simulation-testable proposition and mapped to specific **components, parameters, and figures**. This is the backbone of the design: every component in §3 onward **serves the test of one of these claims**.

| Theoretical claim | Translated into a testable proposition | Corresponding simulation component | Section / figure | Answer obtained |
|---|---|---|---|---|
| **Centola-Macy ①**: information=simple / behaviour=complex | Does the winning mechanism split **by content type**: does IC win for information-like (science news) and threshold win for behaviour-like (rumor = a social judgment to believe and re-spread)? | **Model selection** per dataset×label (IC vs threshold, minimum KS). *Note: the engine itself does not distinguish information/behaviour — see reading notes.* | §3.1-3.2, §3.6-3.7 / F2·F3 | **It did not split**: the winner was decided by the **substrate**, not the content (§4.1). ① could not be confirmed with this data (limitation ⑥) |
| **Centola-Macy ②**: weak ties accelerate simple, impede complex | Does an asymmetry appear: on substrates with **many weak ties** simple runs away, on substrates with **few** complex grows? | **Switching the substrate network**: BA (hubs = many weak-tie bridges) vs WS (high clustering, few bridges) | §3.3 / F2·F3 | Exactly a flip (mechanism in §4.2) |
| **Watts-Dodds ①**: large cascades come from critical mass, not influentials | Is there a region where **hub seeding (influencers)** and **random seeding (the crowd)** swap in reach? | **Seeding experiment** hub vs random, and the **flip boundary** over the (p,φ) plane | §3.4 / F4·F5 | Hubs dominate under single-seed (with the caveat in limitation ⑤) |
| **Watts-Dodds ②**: driven by "easily-influenced" people | How does the presence/absence of cascades change as "susceptibility" goes up or down? | The threshold model's **φ is literally the "resistance to influence" knob** (small φ = easily influenced) | §3.2 / F5 | Large cascades only in the low-φ regime (vanish at the top, φ=0.4, in F5) |

**Reading notes**:
- **Centola-Macy ①'s "information/behaviour" is not an engine distinction.** The simulation distinguishes only **mechanisms (IC/threshold)**; nowhere in the code is there logic that says "this is information, so use simple." "Information vs behaviour" enters only as a **proxy axis via the data labels** (science news = the information pole / rumor = the behaviour-ish proxy), and is tested **indirectly** by "which mechanism wins per content type." But all our data are retweets/shares; there is no pure "behaviour adoption" (purchase, participation) data (limitation ⑥). Since the winner split by substrate rather than content, ① **could not be confirmed** with this data and method — itself an honest reporting point.
- **Centola-Macy ② is tested by "switching the substrate."** On BA (many bridges) simple contagion accelerates and runs away, failing to produce the "moderate" observed shape; on WS (high clustering) complex contagion gets reinforcement and goes global once ignited — this **asymmetric swap of which mechanism runs away** is the direct manifestation of ② (detail in §4.2).
- **Watts-Dodds is tested by "seeding experiment × flip boundary."** Hub = influencer strategy, random = critical-mass strategy. Threshold φ is exactly the susceptibility knob. F5 is a map over (transmissibility p × resistance φ) of which strategy wins in which world.

→ In short, **the components of §3 are not decoration: each exists to measure one of the propositions above.** The conclusions in §4 fill in the "Answer obtained" column.

---

## 2. What we based it on (data)

`data/processed/cascades_unified.csv` (**20,169 cascades**, validated).
`scripts/preprocess.py` converts four public datasets into a unified table (platform, cascade_id, label, size).

| platform | label | count n | median | max |
|---|---|---|---|---|
| higgs | science_news | 13,199 | 2 | 223,833 |
| twitter15 | false/true/unverified/non-rumor | 1,489 | 5–24 | 136 |
| twitter16 | same | 817 | 9–27 | 165 |
| weibo | false/non-rumor | 4,664 | 8–13 | 4,237 |

- **Higgs**: Twitter spread of the 2012 Higgs-boson discovery announcement (De Domenico et al. 2013). A cascade = a **weakly-connected component (WCC)** of the retweet graph (approximation).
- **Twitter15/16**: rumor-verification datasets (Ma et al. 2017). One cascade = the spread of one source tweet.
- **Weibo**: Chinese Weibo rumor/non-rumor (Ma et al. 2016). For cross-platform robustness.

### Note: the actual shape of the Higgs distribution (extreme bimodality)
Higgs is bimodal: "77% are size-2 cascades that die immediately" + "**a single cascade of 223,833 people**" (the runner-up is 69; this one giant holds 87% of all participation). That maximum is a fusion artifact of the WCC approximation (limitation ① in §6, addressed in §5.2).

Below is the observed-size CCDF of the four datasets (Figure F1). The downward-sloping heavy tail — "small cascades dominate, giant ones are rare" — is the shared structure (only Higgs has a long tail out to 220k on the right).

![F1 observed cascade-size CCDF](results/F1_observed_ccdf.png)

---

## 3. How we investigated it (detailed method)

### 3.0 Overview

```
observed data ------------------> observed size distribution
                                        ^
                                        |  compare via KS distance
                                        v
substrate x model x params
   --(Monte Carlo, N runs)-----> simulated size distribution
                                        |
                                        v
              grid search: vary params, minimize KS
                                        |
                                        v
              best IC  vs  best threshold  ->  winner
```

Module-level process flow (Figure F6):

![F6 workflow diagram](results/F6_workflow.png)

Notation:
- Graph `G = (V, E)`. Nodes `V` = users, edges `E` = ties. `n = |V|`.
- Neighbour set `N(v)`, degree `d(v) = |N(v)|`.
- State is binary: each node is "active" (activated = received/spread the information) or "inactive".
- Seed set `S₀ ⊆ V` = initially active nodes (the origin of spread).
- Random generator `rng` (`numpy.random.Generator`, derived from a seed and injected, for reproducibility).

---

### 3.1 Propagation model ① : simple contagion (IC, Independent Cascade)

**Idea**: "hear it once from one person, catch it with probability `p`." The standard model for colds, rumors, information. Each edge fires **at most once in its lifetime**; on success the neighbour activates.

**Rule (discrete time)**:
- A node `u` newly activated at time `t` gets **one trial against each still-inactive neighbour `v ∈ N(u)`**.
- That trial succeeds with probability `p` (activating `v`), fails with `1−p`. Edge `(u,v)` is consumed by this single trial (no retry).
- Trials are independent across edges. Terminate when no new activations occur.

**Equation**: letting `X_{uv}` be the event that edge `(u,v)` fires,

```
P(X_uv = 1) = p ,   independent per edge, a single lifetime trial
v is eventually active  <=>  there is a path from S0 to v using only fired edges
```

This is equivalent to **bond percolation** keeping each edge with probability `p`. The size `|A∞|` of the final active set `A∞` is the "cascade size".

**Algorithm (BFS-frontier; implemented in [models.py](src/contagion_fit/models.py) `SimpleContagion`)**:
```
active <- S0
frontier <- S0
while frontier not empty:
    next <- {}
    for u in frontier:
        for v in N(u) if v not active:
            if rng.random() < p:      # fire with probability p
                active.add(v); next.add(v)
    frontier <- next
return active                          # size = |active|
```
(The implementation is vectorized: gather all frontier edges at once and fire simultaneously with `rng.random(total) < p`.)

**Analytic checks (verified by tests)**:
- `p = 0`: nothing spreads → size = number of seeds.
- `p = 1`: the **entire connected component** of the seed activates.
- Star graph (1 hub + k leaves), seed the hub, one step: each leaf activates independently with probability `p` → **expected size = 1 + k·p**. `test_simulate.py` confirms a 20,000-run average matches this analytic value.

---

### 3.2 Propagation model ② : complex contagion (fractional threshold)

**Idea**: "act only after seeing several acquaintances already do it." The standard model for new behaviour, fashions, risky adoption (the Watts 2002 family). One contact is not enough; **reinforcement** is required.

**Rule (synchronous update)**:
- Each node `v` has a fixed **fractional threshold `φ ∈ [0,1]`**.
- `v` becomes an activation candidate once the fraction of active neighbours reaches `φ`.
- Candidates actually adopt with probability `p` (`p = 1` is deterministic = the Watts 2002 threshold model).
- Update all nodes synchronously, iterating until no change.

**Equation**: letting `a_t(v) = |{u ∈ N(v) : u active at time t}|`,

```
v is an activation candidate at t+1  <=>  a_t(v) / d(v) >= phi
                                     <=>  a_t(v) >= ceil( phi * d(v) )   (required active neighbours)
candidates adopt with probability p (all of them if p=1)
```

`ceil(.)` is the ceiling. The implementation precomputes each node's required count `need(v) = max(ceil(φ·d(v)), 1)` (when φ>0).

**Algorithm ([models.py](src/contagion_fit/models.py) `ComplexContagion`)**:
```
active <- S0
need(v) <- ceil(phi * d(v))  (at least 1 when phi>0)
repeat:
    counts(v) <- count active neighbours per v in bulk via CSR adjacency
    eligible <- { v : counts(v) >= need(v) and v inactive }
    if p < 1: thin eligible by probability p
    if eligible empty: break
    active <- active ∪ eligible
return active
```

**Analytic checks (verified by tests)**:
- `φ = 0`: threshold is zero for everyone → the whole connected component activates.
- `φ > 1`: threshold unreachable → seeds only.
- On a path graph with `φ = 0.6`: interior nodes have degree 2, requiring `ceil(0.6·2) = 2` active neighbours, but the seed side supplies only one → propagation stalls (seed + 1 neighbour only).

**Essential difference from IC**: IC spreads via **a single fired edge** (local, stochastic percolation). The threshold model requires **several active neighbours simultaneously** (global, reinforcement-dependent). This difference drives the later results.

---

### 3.3 Substrate networks: BA and WS (the structure of the stage)

Propagation happens on top of "who is connected to whom." Two substrates are provided to see whether the conclusion depends on structure ([network.py](src/contagion_fit/network.py)).

#### BA (Barabási–Albert) = scale-free network
**Generation rule (preferential attachment)**: start with a few nodes, add new nodes one at a time. Each new node attaches `m` edges to existing nodes, choosing targets **with probability proportional to degree**:
```
P(new node attaches to existing node i) ∝ d(i)     (the rich get richer)
```
The resulting degree distribution is a **power law** `P(k) ~ k^(−3)`. **A few hubs (very high degree) + many low-degree nodes.** Low clustering. Close to real SNS follower networks. Settings: `n=50,000, m=3`.

#### WS (Watts–Strogatz) = small-world network
**Generation rule**: place n nodes on a ring, connect each to its `k` nearest neighbours (ring lattice). Rewire each edge with probability `β` (to a random distant node):
```
β = 0   -> regular lattice (high clustering, large diameter)
β small -> small world (high clustering AND small diameter)  <- used here
β = 1   -> random graph
```
The result has **near-homogeneous degree (≈ k) with no hubs** and **high clustering** (friends-of-friends are friends = many triangles). Close to real neighbourhood / workplace groups. Settings: `n=50,000, k=6, β=0.1`.

| | BA (scale-free) | WS (small-world) |
|---|---|---|
| Degree distribution | power law (extremely uneven, hubs) | near-homogeneous ≈ k |
| Clustering (triangles) | low | **high** |
| Weak ties (distant bridges) | many | few |
| Analogy | SNS follower network | friend / workplace group |

**Why run both**: theory (Centola & Macy) predicts that the "amount of weak ties" flips the advantage between simple/complex. Whether the result changes between BA (many bridges) and WS (few) is itself a **sensitivity analysis**.

---

### 3.4 Monte-Carlo simulation (building the size distribution)

The models are **stochastic** (IC's edge firings are random; the threshold with `p<1` is random in adoption; and the seed location is random). So one run yields only **a single sample**. **Running many times to build a distribution** is the Monte Carlo.

**One run (one trial)**:
```
1. pick seed(s) (strategy below)
2. run the model to termination
3. record the final active-set size S = |A∞|
```

**Seeding strategies** ([simulate.py](src/contagion_fit/simulate.py) `choose_seeds`):
- `random`: uniformly random `seed_count` nodes (default 1).
- `hub`: the top `seed_count` by degree (the highest-degree node = hub).

**Repeating N times** yields per-run sizes `S₁, S₂, …, S_N` (default `N = n_runs = 1000`). This is the **simulated cascade-size distribution**.

```
simulated size distribution = { S_i }_{i=1..N},   S_i = final size of run i
```

**Expected reach** (a reference quantity) is `E[S] ≈ (1/N) Σ S_i`.

**Parallelism and reproducibility**:
- `concurrent.futures.ProcessPoolExecutor` runs the N runs in parallel; each worker builds the substrate graph once (the large graph is never pickled per task).
- Each run uses an **independent substream** `rng = default_rng([seed, run_index])` derived from `seed`, so results are **bit-for-bit reproducible** regardless of worker count.
- Each grid-search cell also gets a non-overlapping `stream_offset`.

---

### 3.5 Bridging observed and simulated: the CCDF

Sizes are heavy-tailed ("mostly small, occasionally huge"). We view them as a **complementary CDF (CCDF)** on log-log axes:

```
CCDF:  F̄(x) = P(X >= x) = (#{ i : S_i >= x }) / N
```

`F̄(x)` is the fraction of cascades of size at least x. On log-log axes the tail shape (power exponent, cutoff) appears as a line/curve (Figures F1–F2, [data.py](src/contagion_fit/data.py) `ccdf`).

---

### 3.6 The fitting yardstick: two-sample KS distance

We measure the "dissimilarity" between observed and simulated distributions with a single number — the **two-sample Kolmogorov–Smirnov statistic**:

```
D = sup_x | F_obs(x) − F_sim(x) |
```

`F_obs, F_sim` are **empirical CDFs**. `D` is the **largest vertical gap** between the two CDF curves. `D ∈ [0,1]`, **smaller = more similar**.

**Important: compute it on the log10 scale.** On raw sizes the bulk of tiny cascades drowns out tail differences, so we transform to `log10(size)` first:
```
D = ks_2samp( log10(observed_sizes), log10(simulated_sizes) ).statistic
```
([fit.py](src/contagion_fit/fit.py) `ks_distance`, `scipy.stats.ks_2samp`).

**Why KS is robust to outliers**: unlike mean / least-squares, `D` only looks at the shape of the CDF and is bounded in `[0,1]`. A single giant has limited leverage in principle (the basis of the discussion in limitation ① §6).

**Properties (test-verified)**: identical samples → `D=0`, fully separated samples → `D=1`.

---

### 3.7 Grid search and model selection

The parameters (`p`, or `φ,p`) are unknown a priori, so we **search for each model's most favorable parameters** ([fit.py](src/contagion_fit/fit.py) `grid_search`, `compare_models`).

**Grid (production)**:
```
IC:        p   ∈ {0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7}
Threshold: φ   ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.4}   ×   p ∈ {0.5, 1.0}
```

**Procedure**: at each grid point run Monte Carlo → compute KS distance → take the minimizing point as the "best fit":
```
D_IC*    = min over p     KS( obs, sim_IC(p) )
D_thr*   = min over (φ,p) KS( obs, sim_threshold(φ,p) )
```

**Winner decision**:
```
margin = | D_IC* − D_thr* |
margin < 0.02              -> "tie"
else if D_IC* < D_thr*     -> "simple wins"
else                       -> "complex wins"
```

We do this per dataset×substrate, producing the verdict table (Figure F3) and `summary.json`.

---

### 3.8 Scale and validity

- Production: **n=50,000 nodes, N=1,000 runs**, on **both BA and WS substrates**.
- **Parameter-recovery test** ([test_fit.py](tests/test_fit.py)): feed synthetic data generated by IC itself into the grid search and check the true `p` is recovered → guarantees the soundness of the method. All 24 pytest cases pass.

---

## 4. Conclusions

### 4.1 Headline: the model-selection verdict **depends on the substrate network (fragile)**

| Dataset | BA winner (IC dist / threshold dist) | WS winner (IC dist / threshold dist) |
|---|---|---|
| higgs (science news) | tie (0.429 / 0.411) | **simple** (0.428 / 0.580) |
| twitter15 (rumor) | **complex** (0.425 / **0.164**) | **simple** (0.307 / 0.487) |
| twitter16 (rumor) | **complex** (0.425 / **0.239**) | **simple** (0.393 / 0.525) |
| weibo (rumor) | **complex** (0.425 / **0.127**) | **simple** (0.307 / 0.424) |

- **BA (scale-free) substrate**: the threshold (complex) model clearly wins on the three rumor datasets.
- **WS (small-world, high-clustering) substrate**: simple contagion (IC) wins everywhere.
- → **There is no single answer; the structure-dependence itself — the verdict is sensitive to the assumed substrate — is the finding.**

KS-distance verdict heatmaps (Figure F3). Smaller number (darker color) = better fit, ★ = winner. **The ★ swaps sides between the two figures**, which is the essence of the structure-dependence:

| BA substrate (star society) | WS substrate (neighbourhood society) |
|---|---|
| ![F3 KS distance BA](results/F3_distance_heatmap.png) | ![F3 KS distance WS](results_ws/F3_distance_heatmap.png) |

* The IC distances in the table are from the initial grid. The later grid extension (§5.1) slightly improves IC distance in some cells (e.g. ba/weibo 0.425→0.351, ws/higgs 0.428→0.360), but **all verdicts are unchanged**. Also the higgs(BA) tie resolves to complex on the pre-announcement subset (§5.2).

### 4.2 Understanding the mechanism (from Figure F2) — why it flips

Best-fit comparison (Figure F2). Black dots = observed, blue line = IC (simple), red line = threshold (complex). The closer a line follows the dots, the better. The thing to watch: **the "bimodal line that runs flat across and departs from the observed" swaps between blue (IC) on BA and red (threshold) on WS**:

| BA substrate (star society) | WS substrate (neighbourhood society) |
|---|---|
| ![F2 best fit BA](results/F2_best_fit_comparison.png) | ![F2 best fit WS](results_ws/F2_best_fit_comparison.png) |

A single-seed model **becomes bimodal** depending on the substrate (either dies `S≈1` or goes nearly global `S≈n`).

- **On BA, IC is bimodal**: ignite a hub and it goes global at once. Near the optimal p about 40% of runs go global, the rest die. With this bimodal shape, **the KS-distance lower bound is essentially set by the "global-cascade probability"**, so no p brings it below ~0.35–0.44 (under the initial narrow grid all three datasets pinned at the boundary p=0.2 with the identical value 0.425; post-extension values in §5.1).
  Meanwhile the threshold model has **hubs themselves hard to activate because of their high degree** (the required active-neighbour count `ceil(φ·d)` is large) = **hubs act as a brake**, reproducing the broad observed tail → complex wins on rumors.
- **On WS, the threshold model becomes bimodal instead**: high clustering supplies reinforcement, so it either goes global once ignited or dies. It cannot reproduce the observed "moderate" shape, while IC follows the middle → simple wins everywhere.

This is **consistent in direction** with Centola & Macy ②: **on BA (many weak-tie bridges) simple contagion accelerates and runs away, while complex is suppressed (blocked by high-degree hubs); on densely-clustered WS complex gets reinforcement and goes global once ignited.** But note the twist in the fitting context: "**the mechanism that is promoted on a given substrate runs away and fails to reproduce the observed 'moderate' shape — and therefore loses.**" The theory's "ease of spreading" and the fit's "closeness to the observed" are different things.

### 4.3 Seeding strategy (Figures F4·F5)
- **BA**: hub seeding wins decisively (threshold model: hub ≈ 50,000 vs random ≈ 0).
- **WS**: at low p nothing spreads regardless of seed; at high p, low φ hubs win.
- **Neither substrate shows a "random-advantage" region.** Under single-seed mean reach a hub never underperforms a random node. A rigorous test of Watts-Dodds' distributed-seeding advantage needs **multiple seeds + cost normalization** → next step.

Seeding experiment (Figure F4, BA substrate) — expected reach of hub vs random seeding:

![F4 seeding BA](results/F4_seeding.png)

Flip-boundary heatmap (Figure F5). Red = hub advantage, blue = random advantage. Over the (transmissibility p × resistance φ) plane, blue (random advantage) **never appears on either substrate**:

| BA substrate | WS substrate |
|---|---|
| ![F5 flip boundary BA](results/F5_flip_boundary.png) | ![F5 flip boundary WS](results_ws/F5_flip_boundary.png) |

---

## 5. Robustness checks

### 5.1 Grid convergence (resolving IC's boundary pinning)
Initially IC's best always stopped at the grid boundary (p=0.2), so we extended the IC grid upward (0.3, 0.5, 0.7) and re-ran ([scripts/grid_robustness.py](scripts/grid_robustness.py)):
- **In all 8/8 cells the IC optimum is now an interior minimum** (a clear U-shaped distance curve, e.g. ba/twitter15: 0.90→**0.44**→0.99). The boundary pinning was **merely a too-narrow grid**.
- Some IC distances improved (ba/weibo 0.425→**0.351** at p=0.15, ws/higgs 0.428→**0.360** at p=0.15), but never enough to overturn the gap to the threshold side.
- **Verdicts unchanged in 7/8 cells** (the one that moved, ba/higgs, is a tie↔complex near-tie).
- → **Even with IC fairly converged, the main conclusion (substrate-dependent flip) holds.**

### 5.2 Higgs pre-announcement subset (removing the WCC-fusion artifact)
We address the influence of the Higgs giant (223,833 people = a measurement artifact) not by deleting an outlier but by **removing the cause** ([scripts/higgs_pre_analysis.py](scripts/higgs_pre_analysis.py)). We rebuild cascades from retweets **before the announcement (2012-07-04), i.e. only 1–3 July 2012**:
- From 30,810 pre-announcement RTs (8.7% of the total), 2,225 cascades are produced.
- **The largest cascade shrinks 223,833 → 21,422** (fusion is partly underway even before the announcement, but greatly reduced). This now **fits within the 50k substrate**, easing the finite-size problem.

**Re-fit results (n=50,000, N=1,000; Figure F7, `results/higgs_pre_summary.json`)**:

| | Full Higgs | Pre-announcement Higgs | Change |
|---|---|---|---|
| BA substrate | tie (IC 0.429 / Thr 0.411) | **complex** (IC 0.435 / **Thr 0.381**) | tie → resolves to complex |
| WS substrate | simple (IC 0.428 / Thr 0.580) | **simple** (IC **0.360** / Thr 0.580) | unchanged |

Pre-announcement comparison figure (Figure F7). Left: observed full vs pre (giant tail shrinks 220k→21k). Center/right: BA/WS best fits to the pre data:

![F7 Higgs pre-announcement robustness](results/F7_higgs_pre.png)

**Interpretation (important)**:
- Once the artifact is removed, the BA "tie" **resolves toward complex**, and Higgs falls into the **same pattern as the rumor data** (BA→complex / WS→simple). So the pre-announcement subset **does not overturn the main conclusion (substrate-dependent flip) — it reinforces it.** The tie was partly a product of the artifact.
- However **a giant component of 21,422 still remains even pre-announcement**, showing WCC fusion is not caused by the announcement burst alone (an honest additional finding). A full fix needs root-level splitting using the timestamped raw trees (Ma's distribution) → next step.

---

## 6. Methodological limitations (stated honestly)

1. **Higgs WCC approximation**: tweet IDs are anonymized, so an exact "one tweet = one cascade" split is impossible. The WCC approximation overestimates size (max 223,833 is essentially the giant component right after the announcement). → Addressed in §5.2.
2. **Identifiability (equifinality)**: different (model, parameter) pairs can produce similar size distributions. Room to add tree depth (structural virality) as a second discriminating axis.
3. **Finite-size effect**: the largest cascade 223,833 cannot fit on an n=50,000 substrate. Addressed by comparing distribution *shape* via KS on the log scale.
4. **Substrate assumption**: that the main conclusion flips between BA and WS is itself reported as a result (fragile, not robust).
5. **Single-seed constraint**: the seeding conclusion is based on single-seed mean reach. A Watts-Dodds test needs multiple seeds and cost normalization.
6. **Coarseness of the "information vs behaviour" proxy**: all our data are retweets/shares (a form of information spread), with no pure "behaviour adoption" (purchase, protest participation) data. Testing Centola-Macy ① relies on a **label-based proxy** ("science news = the information pole / rumor = behaviour-ish"). Moreover the winner split by substrate, not label (§4.1), so ① is **only partially testable**. A clean test needs behaviour-adoption data (e.g. SocioPatterns-style face-to-face contact + adoption records) → next step.

---

## 7. Where the artifacts live

| Type | Path |
|---|---|
| BA substrate figures/results | `results/` (F1–F6, summary.json) |
| WS substrate figures/results | `results_ws/` (F1–F5, summary.json) |
| Grid robustness | `results/grid_robustness.json` |
| Higgs pre-announcement robustness | `results/higgs_pre_summary.json`, `results/F7_higgs_pre.png` |
| Implementation | `src/contagion_fit/` (8 modules), `scripts/` |
| Tests | `tests/` (24 cases, all passing) |
| Design / data provenance | `DESIGN.md`, `DATA_MANIFEST.md`, `README.md` |

How to run:
```bash
python -m contagion_fit --n-nodes 50000 --n-runs 1000 --substrate ba   # production (BA)
python -m contagion_fit --n-nodes 50000 --n-runs 1000 --substrate ws   # sensitivity (WS)
python scripts/grid_robustness.py --n-nodes 50000 --n-runs 1000        # grid robustness
python scripts/higgs_pre_analysis.py --n-nodes 50000 --n-runs 1000     # Higgs pre-announcement
python -m pytest                                                        # tests
```

---

## 8. Next steps (material for the "next steps" slide; out of scope for this assignment)

- **Convergence figure** (overlay CCDFs for n_runs=500 vs 1000) to visualize Monte-Carlo stability.
- Add **tree depth (structural virality)** as a second discriminating axis to ease equifinality.
- **Multiple seeds + cost normalization** to rigorously test the Watts-Dodds distributed-seeding advantage.
- Extensions to Bayesian inference (ABC), temporal fitting, GNNs, etc.
