# Development history

Each entry records the question, result, and reason for the next step. MAP@12
measures how highly the model ranks the distinct articles a buyer purchases in
the next week. Unless stated otherwise, CV means the two September weeks; the
July/August backtests are reported separately. All local weeks were inspected
and are exploratory. Numbers below summarize the saved development notes;
publication cleanup did not rerun the project data.

## Initial exploration: earlier frozen-ranker policy

These results retain their original evaluation policy. They must not be read as
results from the rolling refits introduced in step 18.

- **1. Candidate fusion:** combine popularity, purchase history, collaborative, metadata, and image neighbors.
  Result: CV ≈0.01964; outer-week MAP 0.02084 versus popularity 0.00898. Metadata/image retrieval weakened the blend.
  Next: remove sources to identify which evidence helps ordering.
- **2. Source removal:** test simpler candidate-source combinations.
  Result: history + popularity reached CV 0.02104 and outer MAP 0.02344; more sources did not guarantee better predictions.
  Next: tune the balance between personal history and popularity.
- **3. History weighting:** increase personal-history weight in reciprocal rank fusion.
  Result: weight 2 reached CV 0.02290 and outer MAP 0.02432; weight 1.5 reached CV 0.02285.
  Next: test whether broader retrieval supplies useful articles that fixed weights cannot order well.
- **4. Candidate coverage:** enlarge the purchase-based candidate pool.
  Result: target recall rose from 17.21% to 18.56%, while fusion MAP fell from 0.01963 to 0.01947.
  Next: retain the broader evidence and learn the ordering instead of relying only on fusion weights.
- **5. Learned ranking:** train a histogram gradient-boosted classifier on candidate features.
  Result: CV reached 0.029048, the largest structural improvement in this phase.
  Next: inspect transfer to the later week and separate repeat purchases from new purchases.
- **6. Later-week transfer:** evaluate the frozen reranker and inspect purchase cohorts.
  Result: outer-week MAP improved by roughly 25% over the previous simple baseline, but new-item discovery remained weak.
  Next: test targeted retrieval for articles beyond the customer's purchase history.
- **7. Directed and visual discovery:** add discovery candidates using related products and image evidence.
  Result: no tested variant beat the incumbent overall.
  Next: test whether other customers' purchase sequences provide stronger discovery signals.
- **8. Basket analogs:** retrieve follow-on articles from similar historical baskets.
  Result: the standalone approach did not beat the 0.029048 incumbent.
  Next: improve contextual ordering with seasonality rather than replacing retrieval.
- **9. Seasonal and lifecycle features:** add calendar demand patterns and item-age signals.
  Result: seasonal CV 0.029728 (+2.34%); lifecycle 0.029570; combined 0.029436.
  Next: retain seasonality and investigate where it helps across customer purchase recency.
- **10. Recency routing:** choose between classifiers using observed last-purchase age.
  Result: CV 0.030078 versus seasonal 0.029728; historical mean 0.025695.
  Next: investigate category retrieval and remaining candidate coverage gaps.
- **11. Category retrieval:** expand recommendations through category-level customer preferences.
  Result: no overall improvement over the recency-routed model.
  Next: audit whether the dominant limitation is retrieval or ranking.
- **12. Model audit:** compare actual ranking with perfect ordering of the available candidates.
  Result: outer candidate-oracle MAP 0.200289 versus actual 0.031154 showed substantial ordering headroom; this pool still could not deliver a tenfold gain.
  Next: test stronger customer–article features and alternative learners.
- **13. Learner and profile comparison:** compare XGBoost classifiers/rankers with richer personal profiles.
  Result: profile classifier CV 0.030184 versus recency-route 0.030078; profiles improved the matched XGBoost classifier by 1.64%; query ranking did not win.
  Next: isolate the profile contribution within the existing sklearn classifier.
- **14. Sklearn profiles:** add group, colour, pattern, and visual affinities to the recency route.
  Result: CV 0.030273 (+0.65%); all four weekly point differences were positive, but every paired interval included zero.
  Next: retain the best point estimate while recording uncertainty and preparing an external benchmark.
- **15. First publication package:** document the retained model and save aggregate comparison evidence.
  Result: no new empirical score; the snapshot preserves the initial policy and its limitations.
  Next: consolidate the runnable pipeline around the retained specification.
- **16. Pipeline consolidation:** remove inactive experiment paths from the main workflow.
  Result: synthetic checks supported the retained implementation; no new project-data measurement.
  Next: export predictions for a Kaggle benchmark.
- **17. First Kaggle submission:** test transfer beyond repeatedly inspected local weeks.
  Result: user-reported public MAP 0.02618 and private MAP 0.02632, below September CV 0.030273.
  Next: correct inconsistent classifier refit timing before further model comparisons.

## Fixed two-week rolling refits

At every origin, both classifiers are newly fitted on the two immediately
preceding seven-day label windows. Each window's candidates and features use
only earlier history. The population, metric, and temporal contract are fixed
in [AGENTS.md](../AGENTS.md); comparisons below use that contract.

- **18. Evaluation contract:** make training timing consistent at CV, outer evaluation, and submission origins.
  Result: added policy metadata and rejection tests for stale cutoffs and invalid label windows; implementation alone supplied no new external score.
  Next: rerun and benchmark this corrected procedure before changing capacity or features.
- **19. Rolling-refit benchmark:** run the profile model with the corrected procedure.
  Result: CV 0.030495; user-reported Kaggle public 0.02681 and private 0.02713. The earlier frozen-policy results remain separate.
  Next: measure whether larger training-buyer samples help without changing the evaluation task.
- **20. Nested sample sizes:** compare 10k, 25k, 50k, half, and all eligible buyers per label week, retaining all their candidate rows.
  Result: CV 0.030495 → 0.031148 at 25k (+2.14%); 50k 0.030958; all 0.030747. More data alone did not improve monotonically.
  Next: use 25k provisionally and add BPR customer–article compatibility.
- **21. BPR compatibility:** add a preference score learned from strictly pre-cutoff purchase interactions.
  Result: CV 0.033723 (+8.27%); historical mean +4.59%; new-purchase MAP +40.2%, while repeat-target recall fell from 67.26% to 64.83%.
  Next: retain BPR and test features that describe repeat timing, customer activity, and item demand.
- **22. Feature packages:** compare repeat/activity, demand/lifecycle, and their combination on top of BPR.
  Result: CV 0.034144 (+1.25%) for repeat/activity, 0.033835 for demand/lifecycle, and 0.034091 combined; repeat/activity improved all four weekly point estimates.
  Next: retain repeat/activity and revisit sample size jointly with classifier capacity.
- **23. Sample size × capacity:** cross 25k/50k buyers with 100 iterations/15 leaves and 200 iterations/31 leaves.
  Result: 50k/expanded won at CV 0.034450 (+0.90%) and historical mean 0.028251 (+2.39%); 25k/expanded 0.033143 and 50k/current 0.033873 were weaker.
  Next: retain 50k/expanded + BPR + repeat/activity; benchmark externally before interpreting the small local gain as transferable.
- **24. Selected-model export:** fit development and final origins separately and export the full customer roster.
  Result: export was prepared and subsequently reported complete by the user; no Kaggle score for this iteration is recorded here.
  Next: compare its external score with the earlier 0.02713 private benchmark; the public pipeline now implements this specification only.

The most recent sample/capacity comparison took about 108.6 minutes and peaked
at 33.3 GiB locally; this is an observed experiment cost, not a guarantee for a
fresh end-to-end run. The full original notes and superseded code are preserved
in a local archive, excluded from the publication package.
