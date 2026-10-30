"""Evaluate the origin-refitted model on the previously inspected outer week.

Only this stage reads test.parquet. Do not tune on this exploratory week.
Save all target ranks to distinguish retrieval coverage from final ordering.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from hm_recommender.metrics import score_predictions
from hm_recommender.policy import EVALUATION_POLICY, validate_ranker_bundle
from hm_recommender.ranking import recommend


INPUT_TEST = Path("data/interim/test.parquet")
INPUT_MODEL = Path("results/latest/models/development/model.joblib")
INPUT_TEMPORAL = Path("results/latest/models/development/temporal.joblib")
INPUT_PROFILES = Path("results/latest/models/development/profiles.joblib")
INPUT_RERANKERS = Path(
    "results/latest/models/development/profile_model.joblib"
)
INPUT_IMAGE_EMBEDDINGS = Path("results/features/image_embeddings.npz")
INPUT_SELECTION = Path("results/latest/model_selection.json")
OUTPUT_REPORT = Path("results/latest/evaluation/report.json")
OUTPUT_DIRECTORY = Path("results/latest/evaluation")
OUTPUT_COMPARISON = Path("results/latest/evaluation/comparison.parquet")
SEED = 42

parser = argparse.ArgumentParser(description=__doc__)
parser.parse_args()
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
selection = json.loads(INPUT_SELECTION.read_text())
if selection["experiment"] != ("bpr_repeat_50k_expanded"):
    raise ValueError("Run make fit to create the selected BPR model first.")
if selection.get("evaluation_policy") != dict(EVALUATION_POLICY):
    raise ValueError("Run make fit to regenerate the fixed evaluation policy.")
bundle = joblib.load(INPUT_RERANKERS)
validate_ranker_bundle(
    bundle, cutoff_exclusive="2020-09-16", path=INPUT_RERANKERS
)
rerankers = bundle["models"]
artifacts = joblib.load(INPUT_MODEL)
artifacts.temporal = joblib.load(INPUT_TEMPORAL)
artifacts.profiles = joblib.load(INPUT_PROFILES)
for filename in (
    "manifest.json",
    "temporal_manifest.json",
    "profiles_manifest.json",
):
    metadata = json.loads((INPUT_MODEL.parent / filename).read_text())
    asof = metadata.get("identity", metadata).get("cutoff_exclusive")
    if asof != bundle["prediction_cutoff_exclusive"]:
        raise ValueError("Incompatible latest candidate/profile source cutoff")
source = joblib.load(INPUT_MODEL.parent / "compatibility.joblib")
if (
    source.get("recipe") != "bpr64_100_repeat_activity_7_28_v1"
    or source["cutoff_exclusive"] != bundle["prediction_cutoff_exclusive"]
    or source["evaluation_policy"] != dict(EVALUATION_POLICY)
    or not np.array_equal(source["article_ids"], artifacts.article_ids)
):
    raise ValueError("Incompatible latest compatibility source")
for model in rerankers.values():
    if (
        model.max_iter != 200
        or model.max_leaf_nodes != 31
        or not {"bpr_score", "pair_purchases"} <= set(model.feature_names_in_)
    ):
        raise ValueError("Incompatible latest ranker specification")
artifacts.compatibility = source

with np.load(INPUT_IMAGE_EMBEDDINGS, allow_pickle=False) as saved:
    if not np.array_equal(saved["article_ids"], artifacts.article_ids):
        raise ValueError(
            f"Catalog article order differs in {INPUT_IMAGE_EMBEDDINGS}"
        )
    artifacts.profiles.embeddings = saved["embeddings"]
    artifacts.profiles.readable = saved["readable"]

test = pd.read_parquet(INPUT_TEST)
# Rebuying one article within the week still creates only one relevant target.
targets = (
    test.groupby("customer_id", observed=True)["article_id"]
    .agg(lambda items: sorted(set(items)))
    .to_dict()
)
# Include holdout buyers with no earlier purchase history.
request_customers = sorted(targets)
name = selection["experiment"]
result_dir = OUTPUT_DIRECTORY / name
result_dir.mkdir(parents=True, exist_ok=True)
metric_batches = []
target_batches = []
for start in range(0, len(request_customers), 5000):
    stop = start + 5000
    # Retain candidates to separate retrieval failures from ranking errors.
    predictions = recommend(
        request_customers[start:stop],
        artifacts,
        source_weights=selection["source_weights"],
        k=12,
        collaborative_neighbor_limit=selection["collaborative_neighbors"],
        seed_limit=selection["seed_limit"],
        history_limit=20,
        popularity_limit=100,
        candidate_limit=selection["candidate_limit"],
        half_life_days=30.0,
        rrf_offset=60.0,
        retain_candidates=True,
        reranker=rerankers["older"],
        recent_reranker=rerankers["recent"],
        recent_history_days=7,
        retain_candidate_ranks=True,
    )
    predictions.to_parquet(
        result_dir / f"predictions_{start:07d}.parquet", index=False
    )
    customer_metrics, target_metrics = score_predictions(
        predictions, targets, artifacts, k=12
    )
    metric_batches.append(customer_metrics)
    target_batches.append(target_metrics)

metrics = pd.concat(metric_batches, ignore_index=True)
target_metrics = pd.concat(target_batches, ignore_index=True)
# History size counts distinct purchased articles, not transaction rows.
metrics["history_cohort"] = pd.cut(
    metrics["history_count"],
    bins=[-1, 0, 4, np.inf],
    labels=["none", "1-4", "5+"],
).astype(str)
# Zero-support articles are unseen; 20 is a descriptive rarity cutoff.
target_metrics["support_cohort"] = pd.cut(
    target_metrics["training_support"],
    bins=[-1, 0, 20, np.inf],
    labels=["unseen", "1-20", "21+"],
).astype(str)
metrics.to_parquet(result_dir / "customer_metrics.parquet", index=False)
target_metrics.to_parquet(result_dir / "target_metrics.parquet", index=False)
metrics.groupby("history_cohort", observed=True).agg(
    customers=("customer_id", "size"),
    map_at_12=("ap_at_12", "mean"),
    candidate_recall=("candidate_recall", "mean"),
    recall_at_12=("recall_at_12", "mean"),
).reset_index().to_parquet(
    result_dir / "customer_cohorts.parquet", index=False
)

# Macro recall weights articles equally, limiting popular-item dominance.
item_metrics = (
    target_metrics.groupby(
        ["support_cohort", "repeat_purchase", "article_id"], observed=True
    )
    .agg(item_recall_at_12=("hit_at_12", "mean"))
    .reset_index()
)
# Separate repeat targets from articles new to that customer.
cohorts = (
    target_metrics.groupby(
        ["support_cohort", "repeat_purchase"], observed=True
    )
    .agg(
        targets=("article_id", "size"),
        articles=("article_id", "nunique"),
        hits=("hit_at_12", "sum"),
        retrieved=("retrieved", "sum"),
        target_recall_at_12=("hit_at_12", "mean"),
        candidate_recall=("retrieved", "mean"),
    )
    .reset_index()
)
macro = (
    item_metrics.groupby(["support_cohort", "repeat_purchase"], observed=True)[
        "item_recall_at_12"
    ]
    .mean()
    .reset_index()
)
cohorts = cohorts.merge(macro, on=["support_cohort", "repeat_purchase"])
cohorts.to_parquet(result_dir / "target_cohorts.parquet", index=False)


# Fixed-prediction customer bootstrap: it excludes selection/refit uncertainty.
interval = bootstrap(
    (metrics["ap_at_12"].to_numpy(),),
    np.mean,
    confidence_level=0.95,
    n_resamples=1000,
    batch=50,
    method="percentile",
    random_state=42,
).confidence_interval
summary = {
    "experiment": name,
    "selected_by_cv": True,
    "customers": len(metrics),
    "map_at_12": float(metrics["ap_at_12"].mean()),
    "map_ci_low": float(interval.low),
    "map_ci_high": float(interval.high),
    "new_target_map_at_12": float(metrics["new_target_ap_at_12"].mean()),
    "recall_at_12": float(metrics["recall_at_12"].mean()),
    "candidate_recall": float(metrics["candidate_recall"].mean()),
    "candidate_oracle_map_at_12": float(
        metrics["candidate_oracle_ap_at_12"].mean()
    ),
}
pd.DataFrame([summary]).to_parquet(OUTPUT_COMPARISON, index=False)
report = {
    "seed": SEED,
    "evaluation_policy": dict(EVALUATION_POLICY),
    "test_start": "2020-09-16",
    "test_end_inclusive": "2020-09-22",
    "selected_experiment": name,
    "selected_ranking_method": selection.get(
        "ranking_method", "profile_recency_route"
    ),
    "reranker_training_report": selection["reranker_training_report"],
    "scoring_models": str(INPUT_RERANKERS),
    "scoring_label_end_exclusive": bundle["prediction_cutoff_exclusive"],
    "training_windows": bundle["training_windows"],
    "evaluation_role": (
        "exploratory: this week informed earlier modelling decisions"
    ),
    "scoring_population": (
        "all local-test buyers, including customers without history"
    ),
    "uncertainty": (
        "customer bootstrap, fixed predictions; "
        "no selection or refit uncertainty"
    ),
    "decision": "report the retained model; do not reselect on this week",
    "summaries": [summary],
}
with INPUT_RERANKERS.open("rb") as stream:
    report["ranker_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
OUTPUT_REPORT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
