# H&M Personalized Fashion Recommendations

Machine learning coursework project for the H&M Personalized Fashion Recommendations Kaggle competition.

A reproducible offline pipeline using purchase-based candidate retrieval and
a sklearn classifier with full-history customer profiles. Frozen ResNet-18
embeddings provide visual affinity features. The active pipeline retains the
highest September CV point estimate from the initial modelling iteration.

## Initial results

Mean article-level MAP@12 over **2–8 and 9–15 September 2020**:

| Model | MAP@12 | Role |
|---|---:|---|
| Recent popularity | 0.006854 | Baseline |
| Purchase history + popularity | 0.022897 | Strong simple baseline |
| Sklearn recency route | 0.030078 | Matched reference |
| XGBoost classifier + profiles | 0.030184 | Earlier comparison |
| Sklearn recency route + profiles | **0.030273** | **Retained implementation** |

![Initial CV results and weekly uncertainty](results/published/initial_iteration/overview.png)

Adding profiles to the sklearn reference improves the point estimate by
**0.65%**, with positive point differences in four validation weeks. Every
weekly paired interval includes zero. All inspected weeks are exploratory;
these scores do not establish independent generalization or leaderboard rank.

The same comparison improves new-purchase MAP by **11.74%**, while repeat-target
recall falls from **67.96% to 66.38%** and no-history customer MAP decreases.
Profiles were tested as a twelve-feature package, so the results do not isolate
the contribution of image features. The larger candidate-ordering gap remains.

The focused comparison took **29 minutes 7 seconds** locally. Its six model
fits took about 1.46 minutes; validation feature extraction and cutoff-cache
loading dominated the run. Peak memory was not measured.

[Published aggregate results](results/published/initial_iteration/README.md)
include weekly metrics, paired intervals, cohort summaries, runtime evidence,
and source checksums. No customer-level data or model binaries are included.

## How the model developed

We first tested source removals and history/popularity weights. Broader retrieval
found more targets but ranked them poorly, motivating a learned classifier on a
fixed pool. Directed discovery and basket-trajectory experiments did not improve
the overall score. Seasonal features, observable recency routing, and individual
profiles produced smaller gains after that structural change.

The [modelling overview](docs/modelling_overview.md) summarizes these experiments,
their outcomes, and the validation checks. The development log preserves the
full evidence. Earlier experiment scripts are archived; `pipeline/03_fit.py`
fits only the retained sklearn model.

## Repository map

| Location | Purpose |
|---|---|
| `pipeline/` | Numbered preparation, feature, fitting, evaluation, export, and reporting stages |
| `src/hm_recommender/` | Shared inference, fitted artifact structures, and metrics |
| `tests/` | Pure-function tests and synthetic end-to-end fixtures |
| `notebooks/` | Two unexecuted diagnostic notebooks, including image processing |
| `results/published/initial_iteration/` | Small aggregate snapshot prepared for this iteration |
| [docs/pipeline.md](docs/pipeline.md) | Data contract, model specification, stage outputs, and reproduction |
| [docs/modelling_overview.md](docs/modelling_overview.md) | Short experiment history and testing overview |
| [docs/publication.md](docs/publication.md) | Exact recommended publication files and local-only exclusions |
| [Development log](recommender_development_log.md) | Experiment questions, evidence, and decisions |
| [Model audit](recommender_model_audit.md) | Statistical objective, candidate oracle, and unresolved limitations |

The earlier coursework notebooks remain local research material. The active
workflow uses the numbered scripts and the two notebooks under `notebooks/`.
Archived prototype code is separate from the active workflow and tests.

## Reproduce

Create the project environment and run from this directory:

```bash
conda env create -f environment_ml2847.yml
conda activate hm-recommender
make check
make test
```

`make check` compiles code and validates notebook schemas. `make test` uses
synthetic data and vectors, without real images or weight downloads.

The data is not redistributed. Place these inputs under `data/full/`:

- `transactions.parquet`: purchase date, customer ID, and article ID.
- `articles.parquet`: article ID and the catalog attributes listed in the guide.
- `sample_submission.parquet`: the complete customer roster.
- `images/AAA/AAAAAAAAAA.jpg`: optional local product images.

With the full local inputs available:

```bash
make split         # Chronological preparation and training diagnostics.
make features      # Frozen ResNet-18 features; first run may download weights.
make fit           # Retained model, six branch fits, four CV weeks.
make report        # Summarize its saved aggregate metrics; no fitting.
```

There is one fitting command. Compatible earlier source caches can be reused;
a fresh run does not require any archived experiment. Logs are under
`results/logs/`. `make report` writes `results/model_report.{json,md}` and never
triggers fitting. The frozen publication snapshot can be read without installing
dependencies or data.

`make evaluate` runs the dependency chain and scores the previously inspected
16–22 September outer week. `make export` additionally refits candidate sources
through 22 September and writes full-customer submission files. Neither is
needed to read the historical snapshot. Evaluation and export use the same
retained sklearn route. Scoring classifiers remain frozen while final candidate
sources are updated with the later available purchases.

The [pipeline guide](docs/pipeline.md) documents the chronological boundaries,
feature definitions, cache behavior, and optional export.
The next proposed structural test is a training-size learning curve with model,
features, and candidates fixed; it is not implemented in this iteration.

## Authors and reuse

Group 12:

- Ana Catarina Velez Mendes
- Federico Maria Brignoli
- Isabella Russo
- Mariana Alexandra Feritas Filipe
- Miguel Parra-Castro

The existing [LICENCE](LICENCE) retains the authors' rights and excludes the
Kaggle dataset from its scope. No new open-source licence is introduced here.

Dataset source: [H&M Personalized Fashion Recommendations on Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).
