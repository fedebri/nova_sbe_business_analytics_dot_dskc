"""Summarize the selected model's saved aggregates without loading user data.

CV is optional and never triggered by reporting. Old frozen-model snapshots
remain separate; their scores are not relabelled as rolling-refit results.
"""

import json
from pathlib import Path

import pandas as pd

from hm_recommender.policy import EVALUATION_POLICY

ROOT = Path("results/latest")
selection_path = ROOT / "model_selection.json"
selection = json.loads(selection_path.read_text())
if selection["experiment"] != "bpr_repeat_50k_expanded" or selection[
    "evaluation_policy"
] != dict(EVALUATION_POLICY):
    raise ValueError("Run make fit for the selected model and fixed policy")
report = {
    "schema_version": 2,
    "experiment": selection["experiment"],
    "evaluation_policy": dict(EVALUATION_POLICY),
    "weekly": [],
    "means": [],
    "interpretation": "Previously inspected weeks; exploratory evidence",
}
# A development-only refit must not make old CV output look newly measured.
if selection.get("cv_evaluated", False):
    scores = pd.read_parquet(ROOT / "cv_scores.parquet")
    if not scores["experiment"].eq(selection["experiment"]).all():
        raise ValueError("CV results belong to another model")
    report["weekly"] = json.loads(scores.to_json(orient="records"))
    report["means"] = json.loads(
        scores.groupby("validation_role")
        .agg(weeks=("fold", "size"), map_at_12=("map_at_12", "mean"))
        .reset_index()
        .to_json(orient="records")
    )
evaluation_path = ROOT / "evaluation/report.json"
if (
    evaluation_path.exists()
    and evaluation_path.stat().st_mtime >= selection_path.stat().st_mtime
):
    evaluation = json.loads(evaluation_path.read_text())
    if (
        evaluation["selected_experiment"] == selection["experiment"]
        and evaluation["evaluation_policy"] == dict(EVALUATION_POLICY)
        and evaluation["scoring_label_end_exclusive"] == "2020-09-16"
    ):
        report["outer_week"] = evaluation["summaries"][0]
lines = [
    "# Selected recommender results",
    "",
    f"Model: `{selection['experiment']}`.",
    "",
    "Seven-day buyer-mean MAP@12; two-week rolling classifier refits.",
    "All reported weeks were previously inspected.",
    "",
]
if report["weekly"]:
    lines += ["| CV origin | Buyers | MAP@12 |", "|---|---:|---:|"]
    for row in report["weekly"]:
        lines.append(
            f"| {row['fold']} | {row['scored_customers']:,} | "
            f"{row['map_at_12']:.6f} |"
        )
else:
    lines.append(
        "CV was not replayed in the current fit. "
        "See docs/development_history.md for recorded historical results."
    )
if "outer_week" in report:
    lines += [
        "",
        "Exploratory outer-week MAP@12: "
        f"{report['outer_week']['map_at_12']:.6f}.",
    ]
(ROOT / "model_report.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n"
)
(ROOT / "model_report.md").write_text("\n".join(lines) + "\n")
print(json.dumps(report, indent=2, allow_nan=False))
