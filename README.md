# AlphaOT

AlphaOT compares East Asian and European fine-mapped variants using two AlphaGenome-derived views: regulation and RNA-to-Reactome pathway affinity. It integrates the views by their maximum and solves one global, capacity-constrained unbalanced optimal transport problem per cancer, with lambda = 0.30.

## Reproduce

Use Python 3.13.5 for the recorded environment. No API key, model download, or network access is required after dependency installation.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

The command recomputes all three cancers from the frozen scalar scores, verifies the equations and capacity accounting, and writes:

```text
build/results/       Recomputed result tables and provenance
build/figures/       Individual retained panels in PDF and PNG only
```

The calculation refuses to overwrite nonempty result directories. For another full run, use `python reproduce.py --output build-repeat`. To redraw from an existing validated run, use `python reproduce.py --plots-only`.

## Contents

| Path | Purpose |
|---|---|
| `src/alphaot/` | Frozen scoring, input validation, and global transport code. |
| `config/analysis.json` | Final lambda, track contexts, and pathway resource universe. |
| `inputs/` | Retained SNV candidates and raw credible-set probabilities. |
| `resources/atlas/` | Frozen precomputed scalar scores, exact track catalog, and source hashes. |
| `resources/reactome_mapping.parquet` | Frozen retained gene-pathway mapping and projection weights. |
| `resources/spatial/` | Native prediction bins used by the retained track panels and their gene models. |
| `scripts/figures.py` | Renderer for retained panels b-g and i-j. |
| `figures/` | Reviewed PDF and PNG exports for retained panels b-g and i-j. |

## Recorded result summary

| Cancer | EAS variants | EUR variants | Candidate pairs | Transported pairs | Transported mass |
|---|---:|---:|---:|---:|---:|
| Breast | 49 | 73 | 3,577 | 12 | 6.366666666 |
| Prostate | 101 | 77 | 7,777 | 11 | 4.903846154 |
| Thyroid | 27 | 9 | 243 | 3 | 0.833333333 |

