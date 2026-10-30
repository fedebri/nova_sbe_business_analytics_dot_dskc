"""Create a chronological local holdout and training-only diagnostic artifacts.

Run from the project directory. Source inputs remain untouched.
Duplicate rows remain purchase events. Evaluation deduplicates relevant
article labels. This stage does not inspect test quality.
"""

import json
from pathlib import Path

import pandas as pd


INPUT_TRANSACTIONS = Path("data/full/transactions.parquet")
INPUT_ARTICLES = Path("data/full/articles.parquet")
INPUT_CUSTOMERS = Path("data/full/sample_submission.parquet")
OUTPUT_TRAIN = Path("data/interim/train.parquet")
OUTPUT_TEST = Path("data/interim/test.parquet")
OUTPUT_ARTICLES = Path("data/processed/articles.parquet")
OUTPUT_CUSTOMERS = Path("data/processed/customers.parquet")
OUTPUT_MANIFEST = Path("results/splits.json")
OUTPUT_ACTIVITY = Path("results/train_daily_activity.parquet")

# A fresh checkout contains no generated directories.
for destination in (
    OUTPUT_TRAIN,
    OUTPUT_TEST,
    OUTPUT_ARTICLES,
    OUTPUT_CUSTOMERS,
    OUTPUT_MANIFEST,
    OUTPUT_ACTIVITY,
):
    destination.parent.mkdir(parents=True, exist_ok=True)

# Purchase dates and IDs define the history available at each cutoff.
transactions = pd.read_parquet(
    INPUT_TRANSACTIONS, columns=["t_dat", "customer_id", "article_id"]
)
transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
transactions["customer_id"] = transactions["customer_id"].astype("string")
# Ten-digit article IDs must match the catalog and image filenames.
transactions["article_id"] = (
    transactions["article_id"].astype("string").str.zfill(10)
)
articles = pd.read_parquet(
    INPUT_ARTICLES,
    columns=[
        "article_id",
        "product_type_name",
        "colour_group_name",
        "graphical_appearance_name",
        "index_group_name",
    ],
)
articles["article_id"] = articles["article_id"].astype("string").str.zfill(10)
# Stable catalog positions link articles to feature and neighbor array rows.
articles = articles.sort_values("article_id").reset_index(drop=True)
# Include every submission customer, even those with no training purchases.
customers = pd.read_parquet(INPUT_CUSTOMERS, columns=["customer_id"])
customers["customer_id"] = customers["customer_id"].astype("string")

# Purchases before 16 September are available for development;
# Hold out 16-22 September to test the next week's purchase predictions.
train = transactions.loc[transactions["t_dat"] < "2020-09-16"].copy()
test = transactions.loc[
    transactions["t_dat"].between("2020-09-16", "2020-09-22")
].copy()
train.to_parquet(OUTPUT_TRAIN, index=False)
test.to_parquet(OUTPUT_TEST, index=False)
articles.to_parquet(OUTPUT_ARTICLES, index=False)
customers.to_parquet(OUTPUT_CUSTOMERS, index=False)

# Daily counts help inspect changes in demand within training.
train.groupby("t_dat", observed=True).agg(
    purchase_rows=("article_id", "size"),
    active_customers=("customer_id", "nunique"),
    active_articles=("article_id", "nunique"),
).reset_index().to_parquet(OUTPUT_ACTIVITY, index=False)

# Record dates and counts; Make uses this file to mark preparation complete.
manifest = {
    "train_before": "2020-09-16",
    "test_start": "2020-09-16",
    "test_end_inclusive": "2020-09-22",
    "submission_cutoff": "2020-09-23",
    "target": "distinct article_id purchases in the next seven days",
    "top_k": 12,
    "repeat_purchases_allowed": True,
    "train_rows": len(train),
    "test_rows": len(test),
    "catalog_articles": len(articles),
    "submission_customers": len(customers),
    "test_read_policy": (
        "Only pipeline/04_evaluate.py reads data/interim/test.parquet"
    ),
}
OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
