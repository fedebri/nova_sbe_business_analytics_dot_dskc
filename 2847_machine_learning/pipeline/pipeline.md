# Pipeline and reproduction guide

The six numbered scripts implement one specification:
**50k/expanded + BPR + repeat/activity** (`bpr_repeat_50k_expanded`). Analytical
choices remain visible in the scripts; shared modules implement ranking,
activity aggregation, metrics, and contract validation. No archived experiment
or notebook is required to fit this model from scratch.

## Local inputs

Place these Parquet files under `data/full/`:

| File | Required columns |
|---|---|
| `transactions.parquet` | `t_dat`, `customer_id`, `article_id` |
| `articles.parquet` | `article_id`, `product_type_name`, `colour_group_name`, `graphical_appearance_name`, `index_group_name` |
| `sample_submission.parquet` | `customer_id` for the complete submission roster |

Use the complete purchase history through 22 September 2020. IDs are normalized
as strings, with ten digits for articles. Input files remain unchanged.
The model does not require the original customer-attribute table.

Optional images use `data/full/images/AAA/AAAAAAAAAA.jpg`, where the folder is
the first three digits of the article ID. Missing or unreadable images remain
in the catalog with missing visual evidence. Running without images works but
does not reproduce a model fitted with the full image collection.

Create and activate the environment in `environment_ml2847.yml`. BPR requires
`implicit==0.7.2`, checked before fitting. Run commands from the project root.

## Stages and commands

| Stage | Command | Output |
|---|---|---|
| `01_prepare_data.py` | `make split` | Normalized history, outer labels, catalog/roster, split manifest |
| `02_image_features.py` | `make features` | Frozen image vectors, coverage report, diagnostic neighbors |
| `03_fit.py` | `make fit` | Development sources and classifiers; model selection metadata |
| `03_fit.py --cv` | `make cv` | Optional four-origin CV replay plus development fit |
| `04_evaluate.py` | `make evaluate` | Outer-week predictions, metrics, cohorts, uncertainty summaries |
| `03_fit.py --final` | `make refit` | Separate final sources and classifiers |
| `05_export.py` | `make export` | Full-roster `results/latest/submission.csv` and export manifest |
| `06_report_results.py` | `make report` | Aggregate `results/latest/model_report.{json,md}` |

Make orders preparation → image features → development fit → evaluation → final
refit → export. `make latest-export` is an alias for `make export`, retained for
existing users. Intermediate commands run their prerequisites. Stages rerun
when declared files are newer; progress prints to the terminal.

CV is an explicit replay of the selected specification, not a parameter search.
Ordinary export skips those four historical evaluations. `make report` reads
saved aggregates only; it does not load purchases or classifiers. A
new development-only fit does not label an older CV output as newly measured.
An outer report is included only if its policy/model/cutoff match and it is at
least as recent as model selection.

`make check` compiles Python and validates notebook schemas without executing
cells. `make test` uses synthetic fixtures only. The earlier diagnostic
notebooks are supplementary historical material and may refer to older output
locations; they are not steps in the current pipeline.

## Fixed forecasting and evaluation contract

Predict the next seven calendar days from an **exclusive** history cutoff.
Export twelve distinct valid articles for every customer in the complete
submission roster, preserving leading zeros. Previously purchased articles
remain eligible.

Evaluation averages article AP@12 equally over all next-week buyers, including
customers with no prior history. Non-buyers are excluded. Repeated purchases of
one target article count once; the AP denominator is
`min(number of all distinct target articles, 12)`, including unretrieved targets.
The contract is fixed in [AGENTS.md](../AGENTS.md).

| Training label weeks, inclusive dates | Prediction week | Use |
|---|---|---|
| 8–14 and 15–21 July | 22–28 July 2020 | Optional historical CV |
| 29 July–4 August and 5–11 August | 12–18 August | Optional historical CV |
| 19–25 August and 26 August–1 September | 2–8 September | Optional September CV |
| 26 August–1 September and 2–8 September | 9–15 September | Optional September CV |
| 2–8 and 9–15 September | 16–22 September | Development / outer evaluation |
| 9–15 and 16–22 September | 23–29 September | Final / submission |

Every origin refits both classifiers using precisely
`[cutoff−14 days, cutoff−7 days)` and `[cutoff−7 days, cutoff)`. Candidates,
preprocessing, population priors, BPR, and all other features for a label week
use only purchases strictly before that week's start. Prediction sources use
only purchases strictly before the prediction cutoff.

Only stage 04 reads the prepared outer test file. After that evaluation, stage
03 with `--final` may read the complete raw history, filtering separately for
each source cutoff. The final-fit guard checks the evaluated development
ranker's digest. The two-week training cadence is this project's design choice;
the forecasting task, population, metric, and submission format follow Kaggle.

## Candidate retrieval

- Top 100 articles by distinct buyers in the preceding seven days.
- The customer's latest 20 distinct purchased articles.
- Up to 50 collaborative neighbors for each of five recent purchases.

Collaborative neighbors use cosine similarity on a binary article × customer
matrix from the preceding 90 days. The strongest seed match supplies the source
score, with a 30-day seed half-life. Reciprocal rank fusion combines the three
sources with equal weights and offset 60, retaining at most 500 candidates.
Catalog order breaks ties; popularity fills short lists. BPR supplies a ranking
feature and does not retrieve extra articles.

The serialized source-weight vector retains two trailing zero slots for older
metadata/image sources, preserving compatible cache identities. Those sources
are inactive in the selected recommendation path.

## Ranking features

The **ten base features** describe fusion rank, prior purchase, personal-history
rank and age, distinct history size, last customer purchase age, popularity
rank, item support, collaborative score, and collaborative rank.

**Twelve profile features** describe group share/support, colour lift, pattern
lift, and visual lift/support, each with an equal-weight and a 30-day-decay view.
Profiles use all distinct previously purchased articles, dated by latest
purchase. Groups are `product_type_name × index_group_name`. Population priors
use only pre-cutoff customer/article pairs; categorical priors have one
pseudocount per attribute value and profiles shrink with five prior observations.
Visual profiles use normalized frozen ResNet-18 vectors and a within-group
population prototype with weight five. Visual lift compares candidate affinity
with this prototype. Missing candidate images yield missing affinity; histories
without usable visual evidence have neutral lift and zero support.

**BPR compatibility** comes from binary distinct customer–article purchases
strictly before each cutoff. The CPU implementation uses 64 factors, 100
iterations, learning rate 0.01, regularization 0.01, seed 42, one thread, and
verified negative samples. The feature is the full user/item factor dot product,
including the item-bias coordinate. Unseen users or items receive a missing
score, not an invented preference.

**Ten repeat/activity features** include pair purchase counts and distinct
purchase days; mean gap between distinct purchase dates; elapsed time divided
by that gap; pair counts over 7/28 days; customer purchase days and tenure;
and customer purchase counts over 7/28 days. Purchase counts retain duplicate
transaction rows, while repurchase gaps use distinct dates. Absent counts are
zero; unavailable gaps and ages remain missing. Demand/lifecycle features from
the unsuccessful package comparison are not included.

The **older-history branch adds eight seasonal features**: next-week seasonal
index, strength, confidence, three calendar buckets, customer group share, and
history/season alignment. Monthly category penetration excludes each item's
first 28 selling days and uses completed months ending at least 28 days before
the cutoff. Cyclic smoothing uses `[0.25, 0.5, 0.25]`, with shrinkage for month
coverage and buyer support. Calendar buckets describe observed demand patterns,
not measured temperature or verified launch dates.

## Training and routing

Sample at most **50,000 buyers per label week**, seed 42, from the sorted buyer
list. Keep every candidate for each sampled buyer, including zero-positive
queries; do not inject targets or subsample negatives. Each customer-week has
equal total weight, normalized so the overall mean row weight is one.
Both branches train on the same complete query sample:

```python
HistGradientBoostingClassifier(
    loss="log_loss", learning_rate=0.1, max_iter=200,
    max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0,
    early_stopping=False, random_state=42,
)
```

Only entirely missing columns are dropped. Inference uses the fitted feature
names. Customers with no history or a purchase in the last seven days use the
recent branch; other customers use the seasonal branch. Routing depends on
observable history, never future repeat/new-purchase labels.

## Artifacts, reuse, and interpretation

Image features are extracted once with frozen ImageNet ResNet-18 and official
preprocessing. The first image-bearing run may download weights. Normalized
512-dimensional vectors support visual profiles; within-type neighbors remain
available for the existing image diagnostics, not active candidate retrieval.

Model artifacts live in `results/latest/models/<origin>/`; training reports and
query counts live in `results/latest/training/<origin>/`. Compatible purchase,
profile, and seasonal caches can be reused after cutoff, recipe, input-hash, and
catalog checks. Both ranking classifiers are still refitted at every origin.
Source and feature choices are visible in stage 03, rather than hidden in a
configuration framework.

Ranker bundles store the evaluation policy, prediction cutoff, forecast horizon,
and training windows. `src/hm_recommender/policy.py` rejects incompatible
policies, stale cutoffs, and noncontiguous/future label windows. Evaluation and
export also validate source manifests and BPR/activity compatibility. Export
checks the full roster, catalog membership, and twelve unique articles before
publishing its output files.

CV, outer evaluation, and training reports include local detailed artifacts;
these are not publication files. All local weeks have already been inspected.
Paired uncertainty summaries condition on the fitted predictions and do not
capture model-selection or refitting uncertainty. Historical frozen-ranker
scores remain separate from rolling-refit results; see the
[development history](development_history.md).
