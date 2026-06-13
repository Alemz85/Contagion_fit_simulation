# Plain-language explainer — so what did this project actually do?

> A much simpler, analogy-and-picture version of [FINDINGS_en.md](FINDINGS_en.md).
> Diagrams use **Mermaid** (auto-layout, so no misaligned boxes) plus a couple of ASCII charts kept strictly half-width.
> Japanese versions: [EXPLAINER_ja.md](EXPLAINER_ja.md) / [FINDINGS_ja.md](FINDINGS_ja.md).

---

## 🎯 In one sentence

> **"There are two types of 'going viral'. We looked at real viral data and tried to guess which type it was."**

And what we found —— **"the answer flips depending on how people are connected."**

---

## 1. What did we want to know? — there are two types of "spreading"

Information spreads in roughly two ways.

### Type A: simple contagion (catches fire easily) 🔥

You spread it after hearing it from **just one** person. News, rumors, viruses are this type.

```
  Hear it from one person, and you spread it too. It runs sideways.

  [A] --share--> [B] --share--> [C] --share--> [D] --share--> [E]
```

### Type B: complex contagion (won't move unless everyone is doing it) 🧱

You join only after seeing **several friends** do it. New fashions, joining a protest are this type.

```
  You join only after seeing several friends do it.

  [friend A] --\
  [friend B] ---->  (YOU)  --"saw 3, so I join"-->  [join]
  [friend C] --/
```

> **Question**: are real social-media viral events Type A (simple) or Type B (complex)?
> And does that differ between science news and rumors? Between Twitter and Weibo?

---

## 2. What did we use as material? — 20k real viral events

We used **20,169** real past diffusion records.

| Data | Content | Count |
|---|---|---|
| 🔬 Higgs | "Higgs boson discovered!" science-news spread | 13,199 |
| 🐦 Twitter15/16 | rumors (true/false/unverified/non-rumor) | 2,306 |
| 💬 Weibo | Chinese-SNS rumor/non-rumor | 4,664 |

Each record is "**one viral event = how many people joined the spread**" (the cascade size).

---

## 3. How did we investigate? — a "guess the dish" game

Compare the "real viral event" with our "home-made simulation" and answer with whichever type looks more similar.

```
  real viral event  <----- how similar? (KS distance) ----->  simulation (A or B)
```

The whole procedure:

```
  1. Make fake viral events (both Type A and Type B)
          |
  2. Score how similar to the real one (= KS distance)
          |
  3. The more-similar type wins
          |
  4. Repeat over 4 datasets -> verdict table
```

---

## 4. How do we measure "similarity"? — KS distance

Viral events have a shape: "**mostly small, occasionally gigantic**". Draw the real one and the simulation on the same graph and measure **where the vertical gap is biggest**. That's the **KS distance**.

```
  ratio
  1.0 |***
      |   ***            *** = real
      |  o   **          ooo = simulation (fake)
      |   o    **
      |    oo    **   <=== biggest gap = KS distance
      |     ooo    ***
  0.0 +--------------------> cascade size (log)
```

- **Small KS (near 0) = very similar = good model** ✅
- **Large KS (near 1) = totally different = bad model** ❌

> Example: "Type A = 0.42", "Type B = 0.13" → the smaller, **Type B, wins**.

### First, the shape of the real data (Figure F1)

The four datasets overlaid. Downward-sloping — the shared shape of "lots of small viral events, very few giants".

![F1 distribution of real viral events](results/F1_observed_ccdf.png)

---

## 5. The crux — we tried two kinds of "stage"

A simulation needs a stage of "**who is connected to whom**". We prepared two connection styles. This is what swings the conclusion.

### Stage BA: star society 🌟 (there are celebrities)

Like a real "follower network". **A few celebrities (hubs), and a majority of ordinary people.**

```
        [o]   [o]
          \   /
   [o]---[ HUB ]---[o]      HUB = celebrity (few)
          /   \             o   = ordinary people (many)
        [o]   [o]
```

### Stage WS: neighbourhood society 🏘️ (everyone equal)

Like a real "local / workplace group". **No celebrities; people cluster locally** (friends-of-friends are friends).

```
   [o]-[o]-[o]-[o]
    | X | X | X |        everyone has a similar number of friends
   [o]-[o]-[o]-[o]        dense locally (many triangles)
```

| | BA: star society | WS: neighbourhood society |
|---|---|---|
| Celebrities (hubs) | **yes** (concentrated) | none (everyone equal) |
| Connection density | uneven | **dense locally** |
| Example | SNS followers | friends / workplace |

---

## 6. Result — it flipped depending on the stage!

### Verdict table (Figure F3) — one per stage, two total

The **darker (purple) = smaller number = more similar = better model**. ★ is the winner of that row.
Because we ran **the same contest twice with different stages**, there are two figures.

**① Verdict on the star society (BA)** — the three rumors get ★ in the right column (complex):

![F3 verdict heatmap BA](results/F3_distance_heatmap.png)

**② Verdict on the neighbourhood society (WS)** — everything gets ★ in the left column (simple):

![F3 verdict heatmap WS](results_ws/F3_distance_heatmap.png)

> 👀 Compare the two: **the ★ swaps wholesale between the left column (simple) and the right column (complex)**.
> This is the essence of "the winner flips with the stage".

Putting the two figures into words 👇

| Data | 🌟 winner in BA society | 🏘️ winner in WS society |
|---|---|---|
| 🔬 science news | tie (→ resolves to complex, see below) | **Type A (simple)** |
| 🐦 rumor Twitter15 | **Type B (complex)** | **Type A (simple)** |
| 🐦 rumor Twitter16 | **Type B (complex)** | **Type A (simple)** |
| 💬 rumor Weibo | **Type B (complex)** | **Type A (simple)** |

> **Same data, yet the winner reverses when you change the stage!** 😲
> (Only science-news on BA is a "tie", caused by Higgs's one "ghost data point" discussed in §9. Fixing that resolves it toward complex, matching the rumors.)

### Why does it reverse? — "which one runs away" swaps (Figure F2)

Below, **black dots are real**, blue line = Type A (simple), red line = Type B (complex). The closer a line hugs the dots, the better the fit. These also come one per stage.

**① Star society (BA)**: the blue line (Type A) runs flat across the top-right and departs from the real data = Type A is bad:

![F2 best-fit comparison BA](results/F2_best_fit_comparison.png)

**② Neighbourhood society (WS)**: now the red line (Type B) departs and the blue line (Type A) hugs the real data = the roles reverse:

![F2 best-fit comparison WS](results_ws/F2_best_fit_comparison.png)

The point: **the "runs flat across = becomes an extreme either/or and can't reproduce reality" line swaps between blue (BA) and red (WS) depending on the stage.** That's the essence of the verdict flip.

```
  [1 seed] --+--> dies out               (~57%)
             |
             +--> mass fire, 50k people  (~43%)
```

On the star society, ignite a celebrity and it spreads to everyone at once. So **Type A becomes extreme** and can't reproduce the real "moderate" shape. Type B has a brake, so it fits moderately → **rumors are won by Type B**.

**On the neighbourhood society (WS), this is reversed.** Now Type B becomes extreme and Type A fits moderately → **Type A wins.**

---

## 7. The most important conclusion 🏆

> ## The answer to "which type" **reverses depending on how people are connected**

- Thinking in a star society (follower network) → **rumors look like complex contagion**
- Thinking in a neighbourhood society → **everything looks like simple contagion**

So you **cannot make a single flat claim** like "information is definitely simple contagion".
Rather, **the discovery is that the answer changes depending on the assumed stage**.

> 🤔 **Our initial guess was wrong.** We expected "science news (information) = simple, rumors (social/behaviour-ish) = complex — split by content". In reality **the 'stage' dominated the result far more than the content type.** It did *not* split cleanly into "science news = simple / rumors = complex" — that too is an honest finding.
> (Also, all our data are retweets/shares; there is no pure "behaviour" data, so the information-vs-behaviour comparison is only approximate — a limitation.)

This came out in the direction predicted by a famous theory (Centola & Macy).

```
  star society (many distant bridges) --> simple runs away & misses reality --> COMPLEX wins
  neighbourhood (dense locally)       --> complex runs away & misses reality --> SIMPLE wins
```

> Note a slightly counter-intuitive mapping: **the type that gains momentum in a given town becomes too extreme to reproduce the real "moderate" shape — and loses.** The theory's "ease of spreading" and the fit's "closeness to reality" are different things.

---

## 8. Bonus experiment: who do you spread from? (Figures F4·F5)

We test whether the result changes when the first seed is placed on **a celebrity** vs **a random person**.

![F4 seeding strategy](results/F4_seeding.png)

```
  seed from a celebrity  --> mass fire ~50k
  seed at random         --> almost dies (a few people)
```

→ In the star society, **spreading from a celebrity was overwhelmingly advantageous.**

Below is a map of how advantageous a celebrity start is across all combinations of (transmissibility × caution) — red = celebrity advantage.

![F5 flip boundary](results/F5_flip_boundary.png)

### But wait — that was an unfair fight (Figure F8) ⚖️

Comparing **1 celebrity vs 1 random person** is rigged: of course the celebrity
wins. The real marketing question is: *with a fixed budget*, should you buy **one
celebrity** or **a crowd of cheap ordinary accounts**? In real life a celebrity
costs far more than a nobody — so we gave every strategy the **same budget** and
made **fame expensive** (a hub costs about as much as ~12 ordinary accounts).

```
  budget = 15 coins
  celebrity:  [HUB] costs 15  -> you can afford 1 seed
  crowd:      [o][o][o][o][o][o][o][o] costs ~1 each -> you afford ~8 seeds
```

We let five strategies compete — celebrity (hub), two "smart targeting" methods,
plain **random crowd**, and a **greedy planner** that uses the simulator to hunt
for the best combination — and scored them on **reach per coin spent**.

![F8 budget comparison — simple spread](results/F8_budget_cost_ic.png)

> 🏆 **For simple spread (news-like), the crowd wins.** The one celebrity reaches
> ~9 people for the whole budget; the same budget buys ~7–8 ordinary accounts that
> reach ~14. Even better, the greedy planner — free to pick anyone — **chooses the
> cheap crowd on its own**, confirming the celebrity is a waste of money here.
> **This is exactly the famous "a distributed crowd beats one big influencer"
> claim** (Watts & Dodds) — the thing the 1-vs-1 test in the chart above could
> never show.

For **complex spread** (where you need several friends before you join), it's the
opposite: you need well-placed seeds to get the fire started, so targeting wins
the raw reach while the random crowd sometimes fizzles. So the honest headline is:
**a cheap crowd beats the celebrity for news-like spreading, but not for
behaviours that need peer reinforcement.**

---

## 9. Just to be sure — is it a trick or a coincidence? 🔍

To check that the conclusion (flips with stage) isn't "luck" or an "unfair comparison", we did two sanity checks.

### Check ①: we tried every "best parameter" for each type (the comparison is fair)

To prevent "Type A lost only because we picked a bad setting", we **search a wide range of each type's settings (transmissibility p, caution φ) and pit the best-fitting settings against each other**. At first the grid was narrow and Type A was capped at the edge, so we **widened the range and rechecked** → the **verdicts still didn't change** (same in 7 of 8 cases, the remaining one a near-tie).

### Check ②: remove Higgs's one "ghost data point" and recheck (Figure F7)

Higgs has **a single gigantic cascade of 223,833 people**, which was a **measurement artifact** (many separate spreads appear glued into one right after the announcement) — not one real spread.

So instead of "deleting it as an outlier", we **cut off the cause**: rebuild Higgs using **only the data before the announcement (1–3 July)** and re-measure the KS distance:

![F7 rebuilding Higgs pre-announcement](results/F7_higgs_pre.png)

- The ghost (220k) shrinks to 21k, a size that actually fits on the stage.
- The science-news "tie" on BA then **resolves toward complex**, joining the **same "flips with stage" pattern** as the rumors.
- → The conclusion is **not overturned — it is reinforced.** (Note honestly: a 21k blob remains even before the announcement, so the distortion isn't fully gone.)

---

## 10. All in one picture (workflow diagram F6)

The flow of this project.

![F6 workflow](results/F6_workflow.png)

---

## 🍱 Ultimate summary (an analogy)

> **The same "flammability" spreads completely differently depending on whether you place it at a "fireworks venue" or in a "residential area".**
>
> This project placed the fire of online virality on two kinds of town (star society / neighbourhood society) and found that the conclusion about "what kind of fire it was" flipped depending on the town.
>
> **So to decide "how to make something go viral", you first have to figure out what kind of "town" (connections) your product spreads in.**
> That's the business takeaway.

---

### For those who want more detail
- Precise numbers and methods → [FINDINGS_en.md](FINDINGS_en.md)
- Design rationale → [DESIGN.md](DESIGN.md)
- How to run / code → [README.md](README.md)
