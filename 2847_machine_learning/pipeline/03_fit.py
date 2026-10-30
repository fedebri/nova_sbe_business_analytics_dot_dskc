"""Fit the selected 50k-buyer BPR + repeat/activity recommender.

Each origin trains two classifiers on its own two completed label weeks.
Routing uses observable purchase age, never concealed target repeat status.
Only 04_evaluate.py reads the outer test set. Every prediction origin refits
the classifiers on its two latest completed label weeks, including --final.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from implicit.cpu.bpr import BayesianPersonalizedRanking

from hm_recommender.activity_features import build_activity_tables
from hm_recommender.metrics import score_predictions
from hm_recommender.policy import EVALUATION_POLICY, validate_ranker_bundle
from hm_recommender.ranking import (
    PROFILE_PRIOR_COUNT,
    ProfileArtifacts,
    RecommenderArtifacts,
    TemporalArtifacts,
    candidate_features,
    recommend,
)


INPUT_TRAIN = Path("data/interim/train.parquet")
INPUT_TRANSACTIONS = Path("data/full/transactions.parquet")
INPUT_ARTICLES = Path("data/processed/articles.parquet")
INPUT_IMAGE_NEIGHBORS = Path("results/features/image_neighbors.npz")
INPUT_IMAGE_MANIFEST = Path("results/features/image_manifest.json")
INPUT_IMAGE_EMBEDDINGS = Path("results/features/image_embeddings.npz")
OUTPUT_MODELS = Path("results/latest/models")
OUTPUT_CV = Path("results/latest/cv")
OUTPUT_CV_SCORES = Path("results/latest/cv_scores.parquet")
OUTPUT_SELECTION = Path("results/latest/model_selection.json")
OUTPUT_TRAINING = Path("results/latest/training")
SEED = 42
# Version 3 omits unused metadata/image retrieval; purchase sources
# match version 2 and can reuse its cutoff caches.
FIT_RECIPE_VERSION = 3
TEMPORAL_RECIPE_VERSION = 2
PROFILE_RECIPE_VERSION = 1

started = perf_counter()
timings = []
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--final", action="store_true", help="Refit after local evaluation"
)
parser.add_argument(
    "--cv",
    action="store_true",
    help="Also refit and score the four recorded CV origins",
)
args = parser.parse_args()
if args.final and args.cv:
    parser.error("--final and --cv are separate runs")

if version("implicit") != "0.7.2":
    parser.error("Latest model requires implicit==0.7.2")
if args.final:
    evaluation = json.loads(
        Path("results/latest/evaluation/report.json").read_text()
    )
    if (
        evaluation["evaluation_policy"] != dict(EVALUATION_POLICY)
        or evaluation["selected_experiment"] != "bpr_repeat_50k_expanded"
        or evaluation["scoring_label_end_exclusive"] != "2020-09-16"
    ):
        raise ValueError(
            "Evaluate the latest development model before final fitting"
        )
    with (OUTPUT_MODELS / "development/profile_model.joblib").open(
        "rb"
    ) as stream:
        if (
            hashlib.file_digest(stream, "sha256").hexdigest()
            != evaluation["ranker_sha256"]
        ):
            raise ValueError("Latest evaluation is stale; rerun evaluation")


if args.final:
    # Once evaluation is done, 16-22 September can join the submission history.
    events = pd.read_parquet(
        INPUT_TRANSACTIONS, columns=["t_dat", "customer_id", "article_id"]
    )
    events["t_dat"] = pd.to_datetime(events["t_dat"])
    events["customer_id"] = events["customer_id"].astype("string")
    events["article_id"] = events["article_id"].astype("string").str.zfill(10)
    prediction_periods = [("final", "2020-09-23", None)]
else:
    events = pd.read_parquet(INPUT_TRAIN)
    prediction_periods = [("development", "2020-09-16", None)]
    if args.cv:
        # CV replays the selected specification, not an estimator search.
        prediction_periods = [
            ("cv_2020_07_22", "2020-07-22", "2020-07-29"),
            ("cv_2020_08_12", "2020-08-12", "2020-08-19"),
            ("cv_2020_09_02", "2020-09-02", "2020-09-09"),
            ("cv_2020_09_09", "2020-09-09", "2020-09-16"),
            *prediction_periods,
        ]

# Fixed user constraint: each origin uses exactly its two completed weeks.
# Overlapping weeks may reuse source caches, never an older origin's ranker.
fitting_periods = []
for origin, prediction_cutoff, validation_end in prediction_periods:
    for days_before in (14, 7):
        label_start = pd.Timestamp(prediction_cutoff) - pd.Timedelta(
            days=days_before
        )
        label_end = label_start + pd.Timedelta(days=7)
        fitting_periods.append(
            (
                f"reranker_train_{label_start:%Y_%m_%d}",
                label_start.strftime("%Y-%m-%d"),
                label_end.strftime("%Y-%m-%d"),
                origin,
            )
        )
    fitting_periods.append((origin, prediction_cutoff, validation_end, origin))


articles = pd.read_parquet(INPUT_ARTICLES)
# Catalog positions are compact article identifiers inside the model arrays.
article_ids = articles["article_id"].to_numpy(dtype=str)
article_index = pd.Index(article_ids)
# SKU seasonality inherits a product-type/index-group profile. This grouping
# uses catalog metadata only and is identical at every temporal cutoff.
temporal_group_codes, temporal_groups = pd.factorize(
    pd.MultiIndex.from_frame(
        articles[["product_type_name", "index_group_name"]].fillna("missing")
    ),
    sort=True,
)

# Frozen image vectors supply profile features, not a retrieval source.
image_manifest = json.loads(INPUT_IMAGE_MANIFEST.read_text())
with np.load(INPUT_IMAGE_EMBEDDINGS, allow_pickle=False) as saved:
    if not np.array_equal(saved["article_ids"], article_ids):
        raise ValueError(
            f"Catalog article order differs in {INPUT_IMAGE_EMBEDDINGS}"
        )
    profile_embeddings = saved["embeddings"]
    profile_readable = saved["readable"]
with INPUT_IMAGE_EMBEDDINGS.open("rb") as source:
    profile_image_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
profile_colour_codes, profile_colours = pd.factorize(
    articles["colour_group_name"].fillna("__missing__"), sort=True
)
profile_pattern_codes, profile_patterns = pd.factorize(
    articles["graphical_appearance_name"].fillna("__missing__"), sort=True
)

# Retain the feature-stage identity for compatible source-cache reuse.
fit_inputs = (
    INPUT_TRANSACTIONS if args.final else INPUT_TRAIN,
    INPUT_ARTICLES,
    INPUT_IMAGE_NEIGHBORS,
    INPUT_IMAGE_MANIFEST,
)
input_sha256 = {}
for path in fit_inputs:
    with path.open("rb") as source:
        input_sha256[str(path)] = hashlib.file_digest(
            source, "sha256"
        ).hexdigest()

experiment = "bpr_repeat_50k_expanded"
extra_columns = [
    "bpr_score",
    "pair_purchases",
    "pair_days",
    "pair_mean_gap",
    "pair_gap_ratio",
    "pair_purchases_7d",
    "pair_purchases_28d",
    "customer_purchase_days",
    "customer_tenure_days",
    "customer_purchases_7d",
    "customer_purchases_28d",
]
base_columns = [
    "fusion_rank",
    "previously_purchased",
    "history_rank",
    "days_since_purchase",
    "customer_history_count",
    "days_since_customer_purchase",
    "popularity_rank",
    "item_support",
    "collaborative_score",
    "collaborative_rank",
]
seasonal_columns = [
    "seasonal_index_next_week",
    "seasonality_strength",
    "seasonality_confidence",
    "season_bucket_warm",
    "season_bucket_cold",
    "season_bucket_all",
    "customer_group_history_share",
    "history_season_alignment",
]
profile_columns = [
    "profile_group_long_share",
    "profile_group_long_support",
    "profile_colour_long_lift",
    "profile_pattern_long_lift",
    "profile_visual_long_lift",
    "profile_visual_long_support",
    "profile_group_recent_share",
    "profile_group_recent_support",
    "profile_colour_recent_lift",
    "profile_pattern_recent_lift",
    "profile_visual_recent_lift",
    "profile_visual_recent_support",
]


experiment_spec = {
    "name": experiment,
    "ranking_method": "bpr_repeat_activity_recency_route",
    "source_weights": [1.0, 1.0, 1.0, 0.0, 0.0],
    "collaborative_neighbors": 50,
    "seed_limit": 5,
    "candidate_limit": 500,
}
OUTPUT_TRAINING.mkdir(parents=True, exist_ok=True)
weekly_rows = []
seasonal_references = {}
previous_training_group = None
models = {}
training_features = []
training_labels = []
training_weights = []
training_customers = []

for period, cutoff_text, end_text, training_group in fitting_periods:
    phase_started = perf_counter()
    if training_group != previous_training_group:
        models = {}
        training_features.clear()
        training_labels.clear()
        training_weights.clear()
        training_customers.clear()
        previous_training_group = training_group
    cutoff = pd.Timestamp(cutoff_text)
    fitting = events.loc[events["t_dat"] < cutoff]
    model_dir = OUTPUT_MODELS / period
    model_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{period}: fitting rows={len(fitting):,}; cutoff={cutoff.date()}",
        flush=True,
    )
    model_file = model_dir / "model.joblib"
    manifest_file = model_dir / "manifest.json"
    reuse_fit = False
    if model_file.exists() and manifest_file.exists():
        cached = json.loads(manifest_file.read_text())
        reuse_fit = (
            cached.get("fit_recipe_version") in (2, 3)
            and cached.get("period") == period
            and cached.get("cutoff_exclusive") == cutoff_text
            and cached.get("fitting_rows") == len(fitting)
            and cached.get("input_sha256") == input_sha256
            and cached.get("collaborative_lookback_days") == 90
            and cached.get("popularity_lookback_days") == 7
            and cached.get("collaborative_neighbor_count") == 50
        )
        if reuse_fit:
            artifacts = joblib.load(model_file)
            reuse_fit = np.array_equal(artifacts.article_ids, article_ids)
    if reuse_fit:
        print(f"{period}: reusing fitted sources", flush=True)
    else:
        # Summarize each customer's article purchases, retaining repeat-buy
        # counts.
        history_table = (
            fitting.groupby(["customer_id", "article_id"], observed=True)
            .agg(
                last_purchase=("t_dat", "max"), purchase_rows=("t_dat", "size")
            )
            .reset_index()
        )
        # Recent articles become recommendation seeds; counts break recency
        # ties.
        history_table = history_table.sort_values(
            ["customer_id", "last_purchase", "purchase_rows", "article_id"],
            ascending=[True, False, False, True],
        )
        history_table["article_code"] = article_index.get_indexer(
            history_table["article_id"]
        )
        # Measure ages at this cutoff so recency reflects what was known that
        # day.
        history_table["age_days"] = (
            cutoff - history_table["last_purchase"]
        ).dt.days
        histories = {
            str(customer): (
                group["article_code"].to_numpy(dtype=np.int32),
                group["age_days"].to_numpy(dtype=np.float32),
            )
            for customer, group in history_table.groupby(
                "customer_id", sort=False, observed=True
            )
        }
        # Each customer-article pair appears once, so this counts distinct
        # buyers.
        support = history_table.groupby("article_id", observed=True).size()
        item_support = support.reindex(article_ids, fill_value=0).to_numpy(
            dtype=np.int32
        )
        # Count last week's distinct buyers so reorders do not inflate demand.
        recent = fitting.loc[fitting["t_dat"] >= cutoff - pd.Timedelta(days=7)]
        popularity_counts = recent.groupby("article_id", observed=True)[
            "customer_id"
        ].nunique()
        # Keep all articles so popularity can fill short recommendation lists.
        popularity_table = articles[["article_id"]].copy()
        popularity_table["buyers"] = (
            popularity_table["article_id"].map(popularity_counts).fillna(0)
        )
        popularity = popularity_table.sort_values(
            ["buyers", "article_id"], ascending=[False, True]
        ).index.to_numpy(dtype=np.int32)

        # Article-customer matrix: 1 means a purchase within the previous 90
        # days.
        # Each buyer counts once per article, regardless of repeat purchases.
        collaborative_events = fitting.loc[
            fitting["t_dat"] >= cutoff - pd.Timedelta(days=90)
        ].drop_duplicates(["customer_id", "article_id"])
        # Customer codes become columns; catalog positions become article rows.
        customer_codes, customer_levels = pd.factorize(
            collaborative_events["customer_id"], sort=True
        )
        item_codes = article_index.get_indexer(
            collaborative_events["article_id"]
        )

        collaborative_matrix = csr_matrix(
            (
                np.ones(len(item_codes), dtype=np.float32),
                (item_codes, customer_codes),
            ),
            shape=(len(articles), len(customer_levels)),
        )
        # Articles without recent buyers have no collaborative signal to
        # compare.
        active_items = np.flatnonzero(collaborative_matrix.getnnz(axis=1))

        # Fit cosine neighbors only for articles with recent buyers.
        indices = np.full((len(articles), 50), -1, dtype=np.int32)
        similarities = np.zeros((len(articles), 50), dtype=np.float32)
        if len(active_items):
            search = NearestNeighbors(
                metric="cosine", algorithm="brute", n_jobs=1
            )
            search.fit(collaborative_matrix[active_items])
            # Batches avoid constructing the full distance matrix.
            for start in range(0, len(active_items), 256):
                stop = start + 256
                query = active_items[start:stop]
                distances, local_indices = search.kneighbors(
                    collaborative_matrix[query],
                    n_neighbors=min(51, len(active_items)),
                )
                for row, distance, local_index in zip(
                    query, distances, local_indices, strict=True
                ):
                    destinations = active_items[local_index]
                    keep = (destinations != row) & (distance < 1.0)
                    destinations, distance = destinations[keep], distance[keep]
                    # Catalog order resolves equal-distance ties.
                    order = np.lexsort((destinations, distance))[:50]
                    indices[row, : len(order)] = destinations[order]
                    similarities[row, : len(order)] = np.clip(
                        1.0 - distance[order], 0.0, 1.0
                    )
        neighbors = {"collaborative": (indices, similarities)}
        print(
            f"{period}: collaborative neighbor edges={(indices >= 0).sum():,}",
            flush=True,
        )

        artifacts = RecommenderArtifacts(
            article_ids=article_ids,
            histories=histories,
            popularity=popularity,
            neighbors=neighbors,
            item_support=item_support,
        )
        joblib.dump(artifacts, model_file, compress=3)
        manifest = {
            "period": period,
            "seed": SEED,
            "fit_recipe_version": FIT_RECIPE_VERSION,
            "input_sha256": input_sha256,
            "cutoff_exclusive": cutoff_text,
            "fitting_rows": len(fitting),
            "customers_with_history": len(histories),
            "collaborative_lookback_days": 90,
            "popularity_lookback_days": 7,
            "collaborative_neighbor_count": 50,
            "source_names": ["popularity", "history", "collaborative"],
            "image_manifest": image_manifest,
            "history_file": str(fit_inputs[0]),
        }
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

    # Keep temporal features separate from the expensive cached neighbor fit.
    temporal_file = model_dir / "temporal.joblib"
    temporal_manifest_file = model_dir / "temporal_manifest.json"
    temporal_inputs = {
        str(path): input_sha256[str(path)]
        for path in (fit_inputs[0], INPUT_ARTICLES)
    }
    reuse_temporal = False
    if temporal_file.exists() and temporal_manifest_file.exists():
        saved = json.loads(temporal_manifest_file.read_text())
        reuse_temporal = (
            saved.get("recipe_version") in (1, 2)
            and saved.get("input_sha256") == temporal_inputs
            and saved.get("cutoff_exclusive") == cutoff_text
        )
    if reuse_temporal:
        artifacts.temporal = joblib.load(temporal_file)
        print(f"{period}: reusing temporal features", flush=True)
    else:
        first_sale = fitting.groupby("article_id", observed=True)[
            "t_dat"
        ].min()
        # Completed months only, with a 28-day gap before the cutoff.
        reference_end = (
            (cutoff - pd.Timedelta(days=28)).to_period("M").start_time
        )
        if reference_end not in seasonal_references:
            reference = fitting.loc[
                fitting["t_dat"] < reference_end,
                ["t_dat", "customer_id", "article_id"],
            ].copy()
            reference["group_code"] = temporal_group_codes[
                article_index.get_indexer(reference["article_id"])
            ]
            reference["month"] = reference["t_dat"].dt.to_period("M")
            profiles = np.ones((len(temporal_groups), 12), dtype=np.float64)
            confidence = np.zeros(len(temporal_groups), dtype=np.float64)
            if not reference.empty:
                market_buyers = reference.groupby("month")[
                    "customer_id"
                ].nunique()
                # Exclude the first 28 observed selling days so short launch
                # spikes contribute less to the recurring calendar profile.
                mature = reference.loc[
                    reference["t_dat"]
                    >= (
                        reference["article_id"].map(first_sale)
                        + pd.Timedelta(days=28)
                    )
                ]
                group_buyers = mature.groupby(
                    ["group_code", "month"], observed=True
                )["customer_id"].nunique()
                group_support = mature.groupby("group_code")[
                    "customer_id"
                ].nunique()
                for group, counts in group_buyers.groupby(level="group_code"):
                    counts = counts.droplevel("group_code")
                    observed_months = market_buyers.index[
                        market_buyers.index >= counts.index.min()
                    ]
                    penetration = counts.reindex(
                        observed_months, fill_value=0
                    ) / market_buyers.reindex(observed_months)
                    monthly = penetration.groupby(
                        penetration.index.month
                    ).mean()
                    monthly = monthly.reindex(range(1, 13))
                    confidence[group] = (
                        len(set(observed_months.month))
                        / 12
                        * group_support[group]
                        / (group_support[group] + 200)
                    )
                    raw_profile = (
                        (monthly / monthly.mean()).fillna(1).to_numpy()
                    )
                    smooth = (
                        0.25 * np.roll(raw_profile, 1)
                        + 0.5 * raw_profile
                        + 0.25 * np.roll(raw_profile, -1)
                    )
                    profiles[group] = 1 + confidence[group] * (smooth - 1)
                del mature, group_buyers, group_support, market_buyers
            strength = profiles.std(axis=1)
            # Names describe calendar demand, not measured temperature.
            warm = profiles[:, [4, 5, 6, 7, 8]].mean(axis=1)
            cold = profiles[:, [10, 11, 0, 1, 2]].mean(axis=1)
            tilt = (warm - cold) / (warm + cold)
            buckets = np.full(len(profiles), "unknown_mixed", dtype=object)
            supported = confidence >= 0.5
            buckets[supported & (strength <= 0.15)] = "all_season"
            buckets[supported & (strength > 0.15) & (tilt > 0.10)] = (
                "warm_calendar"
            )
            buckets[supported & (strength > 0.15) & (tilt < -0.10)] = (
                "cold_calendar"
            )
            group_table = temporal_groups.to_frame(index=False)
            group_table["seasonality_confidence"] = confidence
            group_table["seasonality_strength"] = strength
            group_table["season_bucket"] = buckets
            for month in range(12):
                group_table[f"month_{month + 1:02d}_index"] = profiles[
                    :, month
                ]
            seasonal_references[reference_end] = (profiles, group_table)
            del reference
        profiles, group_table = seasonal_references[reference_end]
        # Interpolate cyclic month-center profiles over known calendar dates.
        # Only known next-week calendar dates enter; no outcomes are read.
        dates = pd.date_range(cutoff, periods=7)
        phase = (
            dates.month - 1 + (dates.day - 0.5) / dates.days_in_month
        ).to_numpy()
        calendar_values = np.array(
            [
                np.interp(
                    phase,
                    np.arange(-1, 13) + 0.5,
                    np.r_[profile[-1], profile, profile[0]],
                )
                for profile in profiles
            ]
        )
        next_index = calendar_values.mean(axis=1)
        group_table = group_table.copy()
        group_table["seasonal_index_next_week"] = next_index
        group_table.to_parquet(
            model_dir / "seasonal_groups.parquet", index=False
        )
        temporal_items = group_table.iloc[temporal_group_codes][
            [
                "seasonal_index_next_week",
                "seasonality_strength",
                "seasonality_confidence",
            ]
        ].reset_index(drop=True)
        for name, label in [
            ("warm", "warm_calendar"),
            ("cold", "cold_calendar"),
            ("all", "all_season"),
        ]:
            temporal_items[f"season_bucket_{name}"] = (
                group_table["season_bucket"].to_numpy()[temporal_group_codes]
                == label
            ).astype(np.float32)
        temporal_items = temporal_items.astype(np.float32)
        artifacts.temporal = TemporalArtifacts(
            temporal_items, temporal_group_codes
        )
        joblib.dump(artifacts.temporal, temporal_file, compress=3)
        pd.concat(
            [
                articles[["article_id"]],
                temporal_items,
            ],
            axis=1,
        ).to_parquet(model_dir / "temporal_items.parquet", index=False)
        temporal_manifest = {
            "recipe_version": TEMPORAL_RECIPE_VERSION,
            "input_sha256": temporal_inputs,
            "cutoff_exclusive": cutoff_text,
            "fitting_rows": len(fitting),
            "calendar_reference_end_exclusive": str(reference_end.date()),
            "calendar_minimum_observed_item_age_days": 28,
            "calendar_smoothing": [0.25, 0.5, 0.25],
            "category_support_shrinkage": 200,
            "bucket_min_confidence": 0.5,
            "all_season_max_strength": 0.15,
            "warm_cold_min_tilt": 0.10,
            "warm_calendar_months": [5, 6, 7, 8, 9],
            "cold_calendar_months": [11, 12, 1, 2, 3],
            "bucket_groups": group_table["season_bucket"]
            .value_counts()
            .to_dict(),
            "mean_seasonality_confidence": float(
                group_table["seasonality_confidence"].mean()
            ),
            "catalog_items": len(articles),
        }
        temporal_manifest_file.write_text(
            json.dumps(temporal_manifest, indent=2) + "\n"
        )
        print(json.dumps({"period": period, **temporal_manifest}), flush=True)

    # Customer-profile priors use pre-cutoff support; individual profiles are
    # computed on demand from the complete cutoff-time customer history.
    profile_file = model_dir / "profiles.joblib"
    profile_manifest_file = model_dir / "profiles_manifest.json"
    profile_identity = {
        "recipe_version": PROFILE_RECIPE_VERSION,
        "cutoff_exclusive": cutoff_text,
        "history_sha256": input_sha256[str(fit_inputs[0])],
        "catalog_sha256": input_sha256[str(INPUT_ARTICLES)],
        "embeddings_sha256": profile_image_sha256,
    }
    if (
        profile_file.exists()
        and profile_manifest_file.exists()
        and json.loads(profile_manifest_file.read_text())["identity"]
        == profile_identity
    ):
        artifacts.profiles = joblib.load(profile_file)
        print(f"{period}: reusing profile priors", flush=True)
    else:
        priors = []
        for values, levels in (
            (profile_colour_codes, profile_colours),
            (profile_pattern_codes, profile_patterns),
        ):
            prior = np.ones(
                (len(temporal_groups), len(levels)), dtype=np.float32
            )
            np.add.at(
                prior, (temporal_group_codes, values), artifacts.item_support
            )
            prior /= prior.sum(axis=1, keepdims=True)
            priors.append(prior)
        group_weights = csr_matrix(
            (
                artifacts.item_support.astype(np.float32),
                (temporal_group_codes, np.arange(len(articles))),
            ),
            shape=(len(temporal_groups), len(articles)),
        )
        visual_prior = normalize(
            np.asarray(group_weights @ profile_embeddings), norm="l2"
        )
        artifacts.profiles = ProfileArtifacts(
            group_codes=temporal_group_codes,
            colour_codes=profile_colour_codes,
            pattern_codes=profile_pattern_codes,
            colour_prior=priors[0],
            pattern_prior=priors[1],
            visual_prior=visual_prior,
        )
        # Frozen vectors remain in one catalog file, not eleven cutoff caches.
        joblib.dump(artifacts.profiles, profile_file, compress=3)
        profile_manifest_file.write_text(
            json.dumps(
                {
                    "identity": profile_identity,
                    "population_unit": (
                        "distinct pre-cutoff customer/article pairs"
                    ),
                    "profile_prior_count": PROFILE_PRIOR_COUNT,
                    "groups": len(temporal_groups),
                    "embedding_file": str(INPUT_IMAGE_EMBEDDINGS),
                    "readable_images": int(profile_readable.sum()),
                },
                indent=2,
            )
            + "\n"
        )
    artifacts.profiles.embeddings = profile_embeddings
    artifacts.profiles.readable = profile_readable

    timings.append(
        {
            "phase": "cutoff_sources",
            "period": period,
            "seconds": perf_counter() - phase_started,
        }
    )
    if end_text is not None:
        validation = events.loc[
            (events["t_dat"] >= cutoff)
            & (events["t_dat"] < pd.Timestamp(end_text))
        ]
        targets = (
            validation.groupby("customer_id", observed=True)["article_id"]
            .agg(lambda items: sorted(set(items)))
            .to_dict()
        )
        request_customers = sorted(targets)
    pairs = fitting[["customer_id", "article_id"]].drop_duplicates()
    users, user_ids = pd.factorize(pairs["customer_id"], sort=True)
    items, item_ids = pd.factorize(pairs["article_id"], sort=True)
    matrix = csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (users, items)),
        shape=(len(user_ids), len(item_ids)),
    )
    matrix.sort_indices()
    factorizer = BayesianPersonalizedRanking(
        factors=64,
        learning_rate=0.01,
        regularization=0.01,
        iterations=100,
        num_threads=1,
        verify_negative_samples=True,
        random_state=42,
        dtype=np.float32,
    )
    if matrix.nnz:
        factorizer.fit(matrix, show_progress=True)
    mapping = np.full(len(article_ids), -1, dtype=np.int32)
    positions = article_index.get_indexer(item_ids)
    if (positions < 0).any():
        raise ValueError("BPR history includes unknown catalog IDs")
    mapping[positions] = np.arange(len(item_ids))
    compatibility = {
        "cutoff_exclusive": cutoff_text,
        "evaluation_policy": dict(EVALUATION_POLICY),
        "recipe": "bpr64_100_repeat_activity_7_28_v1",
        "article_ids": article_ids,
        "user_ids": pd.Index(user_ids),
        "global_to_bpr": mapping,
        "user_factors": (
            factorizer.user_factors
            if matrix.nnz
            else np.empty((0, 65), dtype=np.float32)
        ),
        "item_factors": (
            factorizer.item_factors
            if matrix.nnz
            else np.empty((0, 65), dtype=np.float32)
        ),
        "tables": build_activity_tables(
            events,
            cutoff=cutoff_text,
            buyers=user_ids,
            windows=(7, 28),
        ),
    }
    artifacts.compatibility = compatibility
    if not period.startswith("reranker_train_"):
        joblib.dump(
            compatibility, model_dir / "compatibility.joblib", compress=3
        )
    del pairs, matrix, users, items, factorizer, compatibility
    if period.startswith("reranker_train_"):
        # Retain all candidates for sampled buyers, including zero-hit queries.
        # Sampling buyers for training does not change validation denominators.
        request_customers = (
            pd.Series(request_customers)
            .sample(
                n=min(50000, len(request_customers)),
                random_state=42,
            )
            .sort_values()
            .tolist()
        )
        phase_started = perf_counter()
        for start in range(0, len(request_customers), 1000):
            stop = start + 1000
            batch = request_customers[start:stop]
            predictions = recommend(
                batch,
                artifacts,
                source_weights=(1, 1, 1, 0, 0),
                k=12,
                collaborative_neighbor_limit=50,
                seed_limit=5,
                history_limit=20,
                popularity_limit=100,
                candidate_limit=500,
                half_life_days=30.0,
                rrf_offset=60.0,
                retain_candidates=True,
            )
            features = candidate_features(
                predictions,
                artifacts,
                collaborative_neighbor_limit=50,
                seed_limit=5,
                half_life_days=30.0,
                include_temporal=True,
                include_profiles=True,
            )[
                base_columns
                + seasonal_columns
                + profile_columns
                + extra_columns
            ]
            labels = np.concatenate(
                [
                    np.isin(
                        article_ids[row.candidate_codes],
                        targets[row.customer_id],
                    )
                    for row in predictions.itertuples(index=False)
                ]
            ).astype(np.uint8)
            counts = predictions["candidate_count"].to_numpy()
            training_features.append(features)
            training_labels.append(labels)
            training_weights.append(
                np.repeat(1.0 / counts, counts).astype(np.float32)
            )
            training_customers.append(
                pd.DataFrame(
                    {
                        "window": period,
                        "cutoff_exclusive": cutoff_text,
                        "label_end_exclusive": end_text,
                        "customer_id": batch,
                        "candidate_count": counts,
                        "target_count": [len(targets[c]) for c in batch],
                        "retrieved_targets": np.add.reduceat(
                            labels, np.r_[0, np.cumsum(counts)[:-1]]
                        ),
                    }
                )
            )
            print(
                f"{period}: prepared {start + len(batch):,} training queries",
                flush=True,
            )
        timings.append(
            {
                "phase": "training_features",
                "period": period,
                "seconds": perf_counter() - phase_started,
            }
        )
        continue

    if not models:
        # Each buyer-week has equal total weight despite varying pool sizes.
        # Both branches see the same rows; serving routes by observed recency.
        features = pd.concat(training_features, ignore_index=True)
        labels = np.concatenate(training_labels)
        weights = np.concatenate(training_weights)
        # Every customer-week has equal total weight, then mean row weight = 1.
        weights *= len(weights) / weights.sum()
        queries = pd.concat(training_customers, ignore_index=True)
        model_records = []
        for branch, columns in (
            ("recent", base_columns + profile_columns),
            ("older", base_columns + seasonal_columns + profile_columns),
        ):
            columns = columns + extra_columns
            fit_started = perf_counter()
            model = HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.1,
                max_iter=200,
                max_leaf_nodes=31,
                min_samples_leaf=100,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=42,
            )
            # Both route branches train on the same full query set.
            model.fit(
                features[columns].dropna(axis="columns", how="all"),
                labels,
                sample_weight=weights,
            )
            models[branch] = model
            model_records.append(
                {
                    "branch": branch,
                    "parameters": model.get_params(),
                    "feature_columns": model.feature_names_in_.tolist(),
                    "fit_seconds": perf_counter() - fit_started,
                }
            )
            print(
                json.dumps({"group": training_group, **model_records[-1]}),
                flush=True,
            )
        report_dir = OUTPUT_TRAINING / training_group
        report_dir.mkdir(parents=True, exist_ok=True)
        queries.to_parquet(
            report_dir / "training_customers.parquet", index=False
        )
        training_report = {
            "experiment": experiment,
            "training_group": training_group,
            "prediction_cutoff_exclusive": cutoff_text,
            "forecast_end_exclusive": end_text
            or (cutoff + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
            "evaluation_policy": dict(EVALUATION_POLICY),
            "seed": SEED,
            "sklearn_version": version("scikit-learn"),
            "customer_weeks": len(queries),
            "candidate_rows": len(labels),
            "positive_rows": int(labels.sum()),
            "zero_positive_queries": int(
                queries["retrieved_targets"].eq(0).sum()
            ),
            "training_windows": queries.groupby("window", sort=False)
            .agg(
                cutoff_exclusive=("cutoff_exclusive", "first"),
                label_end_exclusive=("label_end_exclusive", "first"),
                customers=("customer_id", "size"),
            )
            .reset_index()
            .to_dict(orient="records"),
            "weighting": (
                "equal total weight per customer-week; mean row weight 1"
            ),
            "target_injection": False,
            "negative_sampling": "none",
            "route_recent_max_days_inclusive": 7,
            "cold_start_route": "recent",
            "input_sha256": input_sha256,
            "embeddings_sha256": profile_image_sha256,
            "models": model_records,
        }
        (report_dir / "training.json").write_text(
            json.dumps(training_report, indent=2) + "\n"
        )
        bundle = {
            "models": models,
            "evaluation_policy": dict(EVALUATION_POLICY),
            "prediction_cutoff_exclusive": cutoff_text,
            "forecast_end_exclusive": training_report[
                "forecast_end_exclusive"
            ],
            "training_windows": training_report["training_windows"],
        }
        ranker_path = model_dir / "profile_model.joblib"
        validate_ranker_bundle(
            bundle, cutoff_exclusive=cutoff_text, path=ranker_path
        )
        joblib.dump(bundle, ranker_path, compress=3)
        training_features.clear()
        training_labels.clear()
        training_weights.clear()
        training_customers.clear()
        del features, labels, weights

    if end_text is None:
        continue

    # Score the retained model; the log preserves historical comparisons.
    phase_started = perf_counter()
    result_dir = OUTPUT_CV / period / experiment
    result_dir.mkdir(parents=True, exist_ok=True)
    metric_batches = []
    target_batches = []
    for start in range(0, len(request_customers), 5000):
        stop = start + 5000
        predictions = recommend(
            request_customers[start:stop],
            artifacts,
            source_weights=(1, 1, 1, 0, 0),
            k=12,
            collaborative_neighbor_limit=50,
            seed_limit=5,
            history_limit=20,
            popularity_limit=100,
            candidate_limit=500,
            half_life_days=30.0,
            rrf_offset=60.0,
            retain_candidates=True,
            reranker=models["older"],
            recent_reranker=models["recent"],
            recent_history_days=7,
            retain_candidate_ranks=True,
        )
        predictions.to_parquet(
            result_dir / f"predictions_{start:07d}.parquet", index=False
        )
        metrics, target_metrics = score_predictions(
            predictions, targets, artifacts, k=12
        )
        metric_batches.append(metrics)
        target_batches.append(target_metrics)
    metrics = pd.concat(metric_batches, ignore_index=True)
    target_metrics = pd.concat(target_batches, ignore_index=True)
    metrics.to_parquet(result_dir / "customer_metrics.parquet", index=False)
    target_metrics.to_parquet(result_dir / "target_ranks.parquet", index=False)
    recency = [
        artifacts.histories.get(c, (None, [np.nan]))[1][0]
        for c in metrics["customer_id"]
    ]
    metrics["recency_cohort"] = (
        pd.cut(
            recency,
            [-1, 7, 30, 90, np.inf],
            labels=["1-7", "8-30", "31-90", "91+"],
        )
        .add_categories("none")
        .fillna("none")
    )
    metrics.groupby("recency_cohort", observed=True).agg(
        buyers=("customer_id", "size"),
        map_at_12=("ap_at_12", "mean"),
        new_target_map_at_12=("new_target_ap_at_12", "mean"),
        candidate_recall=("candidate_recall", "mean"),
    ).reset_index().to_parquet(
        result_dir / "recency_cohorts.parquet", index=False
    )
    new_targets = target_metrics.loc[~target_metrics["repeat_purchase"]]
    repeat_targets = target_metrics.loc[target_metrics["repeat_purchase"]]
    summary = {
        "fold": period,
        "experiment": experiment,
        "validation_role": (
            "exploratory_september"
            if period in ("cv_2020_09_02", "cv_2020_09_09")
            else "historical_backtest"
        ),
        "scored_customers": len(metrics),
        "ranker_label_end_exclusive": cutoff_text,
        "ranker_training_report": str(report_dir / "training.json"),
        "map_at_12": float(metrics["ap_at_12"].mean()),
        "new_target_map_at_12": float(metrics["new_target_ap_at_12"].mean()),
        "candidate_recall": float(metrics["candidate_recall"].mean()),
        "candidate_oracle_map_at_12": float(
            metrics["candidate_oracle_ap_at_12"].mean()
        ),
        "new_target_recall_at_12": float(new_targets["hit_at_12"].mean()),
        "repeat_target_recall_at_12": float(
            repeat_targets["hit_at_12"].mean()
        ),
    }
    weekly_rows.append(summary)
    timings.append(
        {
            "phase": "validation_week",
            "period": period,
            "seconds": perf_counter() - phase_started,
        }
    )
    print(json.dumps(summary), flush=True)

if not args.final:
    scores = pd.DataFrame(weekly_rows)
    if args.cv:
        scores.to_parquet(OUTPUT_CV_SCORES, index=False)
    september = (
        scores.loc[scores["validation_role"].eq("exploratory_september")]
        if args.cv
        else pd.DataFrame()
    )
    selection = {
        "experiment": experiment,
        **{k: v for k, v in experiment_spec.items() if k != "name"},
        "seed": SEED,
        "metric": "MAP@12",
        "evaluation_policy": dict(EVALUATION_POLICY),
        "cv_map_at_12": (
            float(september["map_at_12"].mean()) if args.cv else None
        ),
        "cv_evaluated": args.cv,
        "selection_rule": (
            "fixed selected model; no tuning during evaluation or export"
        ),
        "selection_uncertainty": (
            "exploratory; historical frozen-ranker scores use an older policy"
        ),
        "experiments": [experiment_spec],
        "reranker_training_report": str(
            OUTPUT_TRAINING / "development/training.json"
        ),
        "scoring_models": str(
            OUTPUT_MODELS / "development/profile_model.joblib"
        ),
    }
    selection.update(
        selection_rule="fixed 50k_expanded arm from sample-capacity CV",
        selection_run="results/sample_capacity/20261029T191849005435Z",
        ranking_method="bpr_repeat_activity_recency_route",
        buyers_per_label_week=50000,
        max_iter=200,
        max_leaf_nodes=31,
        compatibility_recipe="bpr64_100_repeat_activity_7_28_v1",
        cv_note=(
            "Four CV origins replayed" if args.cv else "Development fit only"
        ),
    )
    OUTPUT_SELECTION.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2), flush=True)
timings.append({"phase": "total", "seconds": perf_counter() - started})
(
    OUTPUT_TRAINING / ("final_timings.json" if args.final else "timings.json")
).write_text(json.dumps(timings, indent=2) + "\n")
