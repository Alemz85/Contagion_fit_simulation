"""contagion-fit: simple vs complex contagion fitted to real cascade data.

Research question
-----------------
Are the cascade-size distributions of real social-media diffusion better
reproduced by *simple* contagion (Independent Cascade: one stochastic trial per
contact) or by *complex* contagion (fractional threshold: activation needs
reinforcement from several active neighbours)? And does the answer change with
content type (science news / true / false / unverified rumor / non-rumor) and
platform (Twitter / Weibo)?

Public surface
--------------
The package is split into single-responsibility modules so that the structure
itself documents the pipeline:

    config       central parameter container (dataclass)
    data         load the unified cascade table, compute CCDFs
    network      build substrate graphs (BA / Watts-Strogatz / real Higgs)
    models       SimpleContagion (IC) and ComplexContagion (threshold)
    simulate     Monte-Carlo cascade-size sampling (parallel, seeded)
    fit          KS distance, grid search, model comparison
    experiments  seeding strategy and hub/random flip-boundary sweeps
    viz          figures F1-F5
"""

from contagion_fit.config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
