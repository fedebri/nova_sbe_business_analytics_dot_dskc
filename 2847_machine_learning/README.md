# H&M Personalized Fashion Recommendations

Predicting which twelve articles each customer will buy in the next week, using
purchase history, product attributes, and frozen image features. This project
started as machine learning coursework for the H&M Kaggle competition and grew
through a series of controlled recommendation experiments.

The published pipeline focuses on one selected model:
**50k/expanded + BPR + repeat/activity** (`bpr_repeat_50k_expanded`). It retrieves
articles from recent popularity, personal history, and similar purchase
patterns, then learns their order with gradient-boosted classifiers. BPR learns
customer–article compatibility; repeat and activity features describe purchase
frequency and timing. Two classifiers serve customers with recent versus older
purchase histories.

## Recorded results

| Rolling-refit specification | September CV MAP@12 |
|---|---:|
| Profile model, 10k buyers per training week | 0.030495 |
| Profile model, 25k buyers | 0.031148 |
| 25k + BPR | 0.033723 |
| 25k + BPR + repeat/activity | 0.034144 |
| **50k + expanded classifier + BPR + repeat/activity** | **0.034450** |

These are recorded means over **2–8 and 9–15 September 2020**, not measurements
from the publication cleanup. The selected model also reached **0.028251** on
the July/August backtests. Its last step improved September MAP by **0.90%**;
using more buyers and larger classifiers together worked better than either
change alone in that comparison.

All local evaluation weeks have been inspected during development, so these
results are exploratory. The latest model scored **0.03107 public / 0.03135 private** on Kaggle.

## Run locally

```bash
conda env create -f environment_ml2847.yml
conda activate hm-recommender
make check
make test
```

Checks compile the code and validate notebook schemas. Tests use synthetic
purchases and vectors; they do not access project data or download image weights.

Supply local Parquet files under `data/full/`: `transactions.parquet`,
`articles.parquet`, and `sample_submission.parquet`. Product images are optional
and use `images/AAA/AAAAAAAAAA.jpg`. The
[pipeline guide](docs/pipeline.md) lists required columns and image behavior.
Data is not redistributed.

```bash
make export       # Prepare → fit → evaluate → final refit → submission.
make cv           # Optional replay of the four recorded validation origins.
make report       # Summarize saved aggregate results, without fitting.
```

Upload **`results/latest/submission.csv`** to Kaggle. The existing
`make latest-export` command remains an alias for `make export`. Progress prints
in the terminal. A fresh image-feature run may download pretrained weights;
full-data fitting requires substantial memory and CPU time.

## Repository map

| Location | Purpose |
|---|---|
| `pipeline/01_…` through `06_…` | Preparation, image features, fitting, evaluation, export, reporting |
| `src/hm_recommender/` | Shared ranking, activity features, metrics, policy validation |
| `tests/` | Synthetic regression tests, including temporal isolation and full export |
| [Pipeline guide](docs/pipeline.md) | Model settings, dates, feature definitions, commands |
| [Development history](docs/development_history.md) | Short account of experiments, results, and next steps |
| [Publication guide](docs/publication.md) | Public files and local-only material |
| `notebooks/` | Earlier diagnostic notebooks; not required to train or export |
| `results/published/initial_iteration/` | Historical aggregate snapshot under the earlier frozen-ranker policy |

The historical snapshot retains its original interpretation. It is not evidence
for the current rolling-refit model. Full research notes and superseded code are
preserved locally outside the active workflow.

## Authors and reuse

Group 12:

- Ana Catarina Velez Mendes
- Federico Maria Brignoli
- Isabella Russo
- Mariana Alexandra Feritas Filipe
- Miguel Parra-Castro

Dataset source: [H&M Personalized Fashion Recommendations on Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).
