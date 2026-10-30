"""Export all-customer predictions and validate their boundary schema.

Candidate sources and classifiers use the same pre-submission cutoff. Rankers
train on its two most recent completed label weeks under the fixed policy.
Downstream consumers read results artifacts without importing stage code.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hm_recommender.ranking import recommend
from hm_recommender.policy import EVALUATION_POLICY, validate_ranker_bundle


INPUT_CUSTOMERS = Path("data/processed/customers.parquet")
INPUT_MODEL = Path("results/latest/models/final/model.joblib")
INPUT_TEMPORAL = Path("results/latest/models/final/temporal.joblib")
INPUT_PROFILES = Path("results/latest/models/final/profiles.joblib")
INPUT_RERANKERS = Path("results/latest/models/final/profile_model.joblib")
INPUT_IMAGE_EMBEDDINGS = Path("results/features/image_embeddings.npz")
INPUT_SELECTION = Path("results/latest/model_selection.json")
OUTPUT_CSV = Path("results/latest/submission.csv")
OUTPUT_PARQUET = Path("results/latest/submission.parquet")
OUTPUT_MANIFEST = Path("results/latest/export_manifest.json")

parser = argparse.ArgumentParser(description=__doc__)
parser.parse_args()
selection = json.loads(INPUT_SELECTION.read_text())
if selection["experiment"] != ("bpr_repeat_50k_expanded"):
    raise ValueError("Run make fit to create the selected BPR model first.")
if selection.get("evaluation_policy") != dict(EVALUATION_POLICY):
    raise ValueError("Run make fit to regenerate the fixed evaluation policy.")
bundle = joblib.load(INPUT_RERANKERS)
validate_ranker_bundle(
    bundle, cutoff_exclusive="2020-09-23", path=INPUT_RERANKERS
)
rerankers = bundle["models"]
customers = pd.read_parquet(INPUT_CUSTOMERS)
if (
    customers.empty
    or customers["customer_id"].isna().any()
    or not customers["customer_id"].is_unique
):
    raise ValueError("Submission roster must contain unique nonmissing IDs")
artifacts = joblib.load(INPUT_MODEL)
valid_articles = set(artifacts.article_ids)
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

# Keep incomplete batches in .partial files until all customers are processed.
temporary_csv = OUTPUT_CSV.with_suffix(".csv.partial")
temporary_parquet = OUTPUT_PARQUET.with_suffix(".parquet.partial")
# String columns preserve customer IDs and article codes with leading zeros.
schema = pa.schema([("customer_id", pa.string()), ("prediction", pa.string())])
coverage = {
    "customers": 0,
    "without_history": 0,
    "without_personalized_candidates": 0,
}
with pq.ParquetWriter(temporary_parquet, schema=schema) as writer:
    # Batch the full roster to bound memory and preserve customer order.
    for start in range(0, len(customers), 5000):
        stop = start + 5000
        batch = customers["customer_id"].iloc[start:stop].tolist()
        # Previously purchased articles remain eligible for repeat demand.
        predictions = recommend(
            batch,
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
            # Omit candidate lists from returned batches to save memory.
            retain_candidates=False,
            reranker=rerankers["older"],
            recent_reranker=rerankers["recent"],
            recent_history_days=7,
        )

        # Every submitted customer needs exactly 12 distinct article IDs.
        for row in predictions.itertuples(index=False):
            if len(row.predictions) != 12 or len(set(row.predictions)) != 12:
                raise ValueError(
                    f"Expected 12 distinct article IDs for {row.customer_id}"
                )
            if any(
                item not in valid_articles
                or len(item) != 10
                or not item.isdecimal()
                for item in row.predictions
            ):
                raise ValueError("Invalid submission article ID")
        exported = predictions[["customer_id"]].copy()
        # Format ranked IDs as the space-separated string expected in the CSV.
        exported["prediction"] = predictions["predictions"].str.join(" ")
        table = pa.Table.from_pandas(
            exported, schema=schema, preserve_index=False, safe=True
        )
        writer.write_table(table)
        # Append batches under one header to keep memory use bounded.
        exported.to_csv(
            temporary_csv,
            index=False,
            mode="w" if start == 0 else "a",
            header=start == 0,
        )
        # Count customers lacking purchase history or personal candidates.
        coverage["customers"] += len(exported)
        coverage["without_history"] += int(
            (predictions["history_count"] == 0).sum()
        )
        coverage["without_personalized_candidates"] += int(
            (~predictions["used_personalization"]).sum()
        )
        print(
            f"Exported {coverage['customers']:,} / {len(customers):,} "
            "customers",
            flush=True,
        )

# Publish the completed outputs before marking this stage complete.
temporary_csv.replace(OUTPUT_CSV)
temporary_parquet.replace(OUTPUT_PARQUET)
# Record dates and coverage; Make uses this file to mark export complete.
manifest = {
    "schema_version": 1,
    "evaluation_policy": dict(EVALUATION_POLICY),
    "columns": {
        "customer_id": "string",
        "prediction": "12 space-separated article_id strings",
    },
    "article_id_format": "ten digits, preserving leading zeros",
    "selected_experiment": selection["experiment"],
    "ranking_method": selection.get("ranking_method", "rrf"),
    "collaborative_neighbors": selection["collaborative_neighbors"],
    "seed_limit": selection["seed_limit"],
    "candidate_limit": selection["candidate_limit"],
    "scoring_models": str(INPUT_RERANKERS),
    "scoring_label_end_exclusive": bundle["prediction_cutoff_exclusive"],
    "training_windows": bundle["training_windows"],
    "ranker_training_report": ("results/latest/training/final/training.json"),
    "route_recent_max_days_inclusive": 7,
    "fit_cutoff_exclusive": "2020-09-23",
    "training_end_inclusive": "2020-09-22",
    "forecast_start": "2020-09-23",
    "forecast_end_exclusive": "2020-09-30",
    "forecast_horizon_days": 7,
    "metric": "MAP@12",
    "coverage": coverage,
    "monitoring": {
        "online_impact": "unverified; this is an offline challenge pipeline",
        "checks": [
            "customer coverage",
            "12 unique valid IDs",
            "fallback frequency",
            "runtime logs",
        ],
    },
}
OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
