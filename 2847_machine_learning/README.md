# H&M Personalized Fashion Recommendations

Machine learning coursework project for the H&M Personalized Fashion Recommendations Kaggle competition.

The project explores customer, article, and transaction data, then compares recent
popularity, purchase-history, collaborative, metadata, and frozen-image signals.

The new pipeline and both diagnostic notebooks are **unexecuted scaffolding**.
No project data or images were processed, no models were trained, and no weights
were downloaded while preparing them. Runtime, memory use, and model quality are
unverified until you run the stages.

## Project Structure

- `pipeline/01_prepare_data.py` - normalized inputs, chronological holdout, and training activity report.
- `pipeline/02_image_features.py` - frozen ResNet-18 embeddings and image neighbors.
- `pipeline/03_fit.py` - rolling time CV, source ablations, and optional final refit.
- `pipeline/04_evaluate.py` - held-out evaluation of the frozen development choice.
- `pipeline/05_export.py` - batched, schema-checked predictions for every submission customer.
- `src/hm_recommender/ranking.py` - fitted artifact type and inference shared by CV, evaluation, and export.
- `src/hm_recommender/metrics.py` - notebook-derived AP@12 and diagnostics shared by CV and evaluation.
- `notebooks/01_recommender_diagnostics.ipynb` - training support, CV, retrieval, and cohort diagnostics.
- `notebooks/02_image_diagnostics.ipynb` - image coverage, preprocessing inspection, and visual neighbors.
- `Makefile` - orders stages and writes execution logs under `results/logs/`.
- `tests/` - pure inference/metric tests and a small synthetic-file integration test.
- `ML_final_project_eda.ipynb` - cleaned exploratory data analysis notebook.
- `ML_final_project_model.ipynb` - modeling notebook reading Parquet inputs from `data/full`.
- `hierarchy_reconstruction.ipynb` - article feature hierarchy and dependency analysis.
- `environment_ml2847.yml` - conda environment definition.

## Data

The Kaggle data is not included in this repository. Place local files under `data/full`:

- `articles.parquet`
- `customers.parquet`
- `sample_submission.parquet`
- `transactions.parquet`

The new pipeline requires articles, transactions, and the submission customer
roster. `customers.parquet` remains available to the earlier exploratory notebooks;
demographic features are not used in this first model.

Place product images at `data/full/images/AAA/AAAAAAAAAA.jpg`, using the first
three digits of each ten-digit article ID as the subfolder. Missing or unreadable
images are recorded and use non-image recommendation sources.

Existing `data/full/` files are treated as read-only raw inputs; they are not moved.
Stage files are written to `data/interim/` and `data/processed/`. Reports, fitted
artifacts, and exports are written to `results/`. These directories are ignored by
git. The fixed split expects the source history to end on **22 September 2020**.
Use the complete inputs for evaluation; the old daily transaction subsampling is
not used by this pipeline.

## Running

Create the environment:

```bash
conda env create -f environment_ml2847.yml
conda activate hm-recommender
```

All following commands run from the repository root:

```bash
make check       # Parse/compile code and validate notebooks; no data execution.
make test        # Small synthetic-file and array tests; no real images or downloads.
make evaluate    # Split -> image extraction -> CV fitting -> held-out evaluation.
```

`make evaluate` reads and processes the real inputs. Its first image-feature run
may download ImageNet ResNet-18 weights using torchvision. The encoder is frozen,
uses the official `IMAGENET1K_V1` preprocessing, and produces normalized
512-dimensional pooled features. CUDA is used if available, otherwise MPS, then
CPU. Image and metadata neighbor searches stay within product type; collaborative
neighbors use shared purchase behavior. Neighbors are precomputed in batches,
without a dense all-customer/item matrix. Neighbor construction and stored customer
histories are the main runtime/memory costs; they have not been benchmarked here.

Inspect logs during a run with, for example:

```bash
tail -f results/logs/03_fit.log
```

Then open the two notebooks with the `hm-recommender` environment selected:

```bash
jupyter lab notebooks/
```

The notebooks only inspect `data/` and `results/`; no pipeline stage depends on
their outputs. The first notebook's support sections need `make split`; its CV
and test-report sections need `make evaluate`. The image notebook's preprocessing
and neighbor sections need `make features`; its ablation section needs evaluation.
Cells are intentionally unexecuted. Findings that change the analysis belong as
literal choices in the relevant stage script, not in notebook-only state.

After reviewing the local evaluation, generate the final challenge predictions:

```bash
make export
```

This first ensures evaluation exists, then invokes `03_fit.py --final` using all
historical purchases through 22 September and writes `results/submission.csv`,
`results/submission.parquet`, and `results/export_manifest.json`. The export
contains 12 distinct article IDs for every submission customer, forecasting
23–29 September 2020. The manifest records the training cutoff, forecast dates,
and coverage counts.

Make uses completion manifests for stage dependencies. If source images change,
run `make -B features` before fitting again. If an output was removed manually,
rerun its stage with `make -B <stage>`. Keep cached artifacts with their manifests;
successful file loading alone is not evidence that they match a new experiment.

### Starting choices and evaluation boundaries

- Predict 12 distinct **article IDs**, preserving leading zeros. Previously
  purchased articles remain eligible. Catalog metadata is permitted benchmark
  input; recency is a demand signal, not a hard stock-availability filter.
- Reserve **16-22 September 2020** for the local test. CV validates on **2-8** and
  **9-15 September**, fitting only on earlier transactions within outer training.
  Only `04_evaluate.py` reads the named local-test file.
- Use recent seven-day distinct-customer popularity, 90-day binary collaborative
  evidence, 20 neighbors per item, and the five most recent history seeds.
- Fuse sources with RRF offset 60, a 30-day recency half-life, and at most 300
  initial candidates. Similarity sources use the strongest recency-weighted seed
  match. Short lists are completed from popularity without replacing ranked items.
- Compare popularity; history+collaborative; add metadata; add images with weight
  0.5 or 1.0. Select the largest mean weekly CV MAP@12, breaking exact ties in
  favor of the earlier, simpler ablation. There is no learned blending gate.
- Evaluate that fixed grid on the local test for diagnosis, without changing the
  selected model. Inspect candidate recall and MAP@12 separately. MAP is not the
  percentage of useful recommendations. Duplicate target purchases count as one
  relevant article; customers without a future purchase are unscored, not zeros.
- Report customer history cohorts and crossed training-support/repeat-target
  cohorts. The illustrative `1-20` support group is **descriptive**, not a learned
  threshold or a transferred acceptance rule. No cohort loss budget is assumed.
- Report paired customer-bootstrap intervals against popularity. These intervals
  concern fixed predictions and do not cover previous tuning or training variation.

Parameters and feature lists are visible in the stage scripts. No configuration
classes or hidden model factories are used. Shared `src/` code exists because the
same inference and metric calculations are needed by multiple stages and merit
unit tests. Downstream reports or sites should read the exported `results/`
artifacts and manifest, without importing pipeline code. The public prediction
schema is validated in `05_export.py`.

### Results to discuss after the first run

Review `results/cv_scores.parquet`, `results/model_selection.json`,
`results/evaluation/comparison.parquet`, the selected model's customer/target cohort
tables, and `results/features/image_manifest.json`. In particular, check whether
images add retrieval coverage or ranking quality, whether repeat and new-item
performance move in opposite directions, and how often personalization is absent.
No online monitoring service or causal user-impact claim is included; export
coverage and fallback counts provide an offline monitoring starting point.

### Earlier exploratory notebooks

The original notebooks are preserved and can be read independently:

1. `ML_final_project_eda.ipynb`
2. `hierarchy_reconstruction.ipynb`
3. `ML_final_project_model.ipynb`

## Authors

Group 12:

- Ana Catarina Velez Mendes
- Federico Maria Brignoli
- Isabella Russo
- Mariana Alexandra Feritas Filipe
- Miguel Parra-Castro

Dataset source: [H&M Personalized Fashion Recommendations on Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).
