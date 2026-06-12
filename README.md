# contagion-fit

**Simple vs complex contagion: which mechanism explains real social-media cascade size distributions?**

This project pits two competing diffusion mechanisms against real information
cascades and asks which one reproduces the observed cascade-size distribution
better — and whether the answer depends on content type and platform.

> **Research question.** Are the cascade-size distributions of real social-media
> diffusion better reproduced by *simple* contagion (Independent Cascade: one
> stochastic trial per contact) or by *complex* contagion (fractional threshold:
> activation needs reinforcement from several active neighbours)? And does the
> answer change with content type (science news / true / false / unverified
> rumor / non-rumor) and platform (Twitter / Weibo)?

Two strands of theory are brought to a common, data-driven test:

- **Centola & Macy (2007):** information spreads by simple contagion, behaviour
  by complex contagion; weak ties accelerate the former and impede the latter.
- **Watts & Dodds (2007):** large cascades are driven not by influentials but by
  a critical mass of easily-influenced individuals.

**Business translation.** If your content spreads by simple contagion,
influencer-seeded campaigns pay off; if it spreads by complex contagion,
seeding clustered groups of ordinary users wins. Diagnosing the type is what
should drive budget allocation.

## Installation

```bash
python -m pip install -e .          # or: pip install networkx numpy scipy matplotlib pytest
```

Python 3.11+ is required (developed and tested on 3.14).

## Running the analysis

```bash
# quick smoke run (small substrate, few Monte-Carlo runs)
python -m contagion_fit

# full run used for the figures
python -m contagion_fit --n-nodes 50000 --n-runs 1000 --substrate ba

# sensitivity check on a different substrate
python -m contagion_fit --n-nodes 50000 --n-runs 1000 --substrate ws
```

Outputs land in `results/` (override with `--results DIR`):

| File | Content |
|------|---------|
| `F1_observed_ccdf.{svg,png}` | observed cascade-size CCDF, all datasets overlaid (log-log) |
| `F2_best_fit_comparison.{svg,png}` | observed vs IC-best vs threshold-best CCDF, per dataset |
| `F3_distance_heatmap.{svg,png}` | KS-distance matrix (datasets x models), winner starred |
| `F4_seeding.{svg,png}` | hub vs random seeding reach, with error bars |
| `F5_flip_boundary.{svg,png}` | hub-advantage surface over the (p, phi) plane |
| `summary.json` | all fit results, winners, and experiment outputs |

## Package layout

The code is split into single-responsibility modules so the structure itself
documents the pipeline:

```
src/contagion_fit/
  config.py       central parameter container (frozen dataclass); the only
                  source of randomness is Config.rng()
  data.py         load the unified cascade table, compute CCDFs
  network.py      build substrate graphs (BA / Watts-Strogatz / real Higgs)
  models.py       SimpleContagion (IC) and ComplexContagion (threshold),
                  one shared, side-effect-free run() interface
  simulate.py     Monte-Carlo cascade-size sampling (serial + parallel, seeded)
  fit.py          KS distance on log10(size), grid search, model comparison
  experiments.py  seeding strategy and hub/random flip-boundary sweeps
  viz.py          figures F1-F5
  __main__.py     end-to-end pipeline
```

All randomness is injected as a `numpy.random.Generator` derived from
`Config.seed`, so every run is fully reproducible regardless of worker count.

## Data

The processed table `data/processed/cascades_unified.csv` (20,169 cascades) is
built by `scripts/preprocess.py` from four public datasets. See
[DATA_MANIFEST.md](DATA_MANIFEST.md) for provenance and validation, and
[DESIGN.md](DESIGN.md) for the full methodology.

| platform | label | n | median | max |
|----------|-------|---|--------|-----|
| higgs | science_news | 13,199 | 2 | 223,833 |
| twitter15 | false / true / unverified / non-rumor | 1,489 | 5–24 | 136 |
| twitter16 | same | 817 | 9–27 | 165 |
| weibo | false / non-rumor | 4,664 | 8–13 | 4,237 |

## Methodological caveats

These are documented honestly (see DESIGN.md section 5) and are part of the
story, not hidden: the Higgs weakly-connected-component approximation
over-estimates cascade size; different (model, parameter) pairs can produce
similar size distributions (equifinality); a finite substrate cannot host the
largest observed cascades, so comparisons use the distribution *shape* via KS on
log sizes. Both BA and WS substrates are run to separate robust from fragile
conclusions.

## Tests

```bash
python -m pytest
```

The suite encodes analytic edge cases (`p=0`, `p=1`, `phi=0`, `phi>1` on path,
star, complete graphs), reproducibility under a fixed seed, the star-graph IC
mean matching the analytic `1 + k*p`, and a **parameter-recovery test**:
synthetic IC data is fed back through the grid search and the true `p` is
recovered, demonstrating the fitting machinery is sound.

## Citations

- De Domenico et al. (2013) "The Anatomy of a Scientific Rumor", *Sci. Rep.* — Higgs data
- Ma et al. (2016, IJCAI / 2017, ACL) — Weibo / Twitter15/16 rumor datasets
- Centola & Macy (2007) *AJS* — complex contagion and weak ties
- Watts & Dodds (2007) *J. Consumer Research* — testing the influencer hypothesis
- Watts (2002) *PNAS* — threshold cascade theory
```
