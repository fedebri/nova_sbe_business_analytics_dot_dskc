# U.S. Flight Network Analysis

Federico Maria Brignoli and Felipe Botelho, 2022.

Portfolio project developed during the MSc in Business Analytics at NOVA School of Business and Economics.

This project analyzes U.S. flight operations using flight records, airport metadata, and U.S. public holiday data. The notebook keeps the original Colab/Spark setup cells for reference, but the showcase version runs locally with pandas, scikit-learn, matplotlib, and NetworkX.

## Contents

- `us_flights_delay_prediction_clustering.ipynb`: end-to-end analysis notebook.
- `data/airports.csv`: airport metadata.
- `data/holidays.csv`: U.S. public holidays.
- `data/sample.csv`: sample flight records used by default.

## Analysis Covered

1. Spark versus pandas aggregation workflow.
2. Top airports, busiest days, time-of-day traffic, airport delay, carrier delay, and period delay statistics.
3. Route-carrier delay indices for traveller-facing lookup use cases.
4. Binary departure-delay classification for `SHORT` and `LONG` delays.
5. Airport clustering and airport-route community detection.

## Running the Notebook

Install the Python dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn networkx seaborn jupyter
```

Then open and run:

```bash
jupyter notebook us_flights_delay_prediction_clustering.ipynb
```

The default notebook expects `data/sample.csv`, `data/airports.csv`, and `data/holidays.csv`. The larger raw data files are not suitable for normal GitHub storage because they exceed GitHub's regular file-size limits; use Git LFS or download them separately if needed.

## Notes

The notebook is intentionally pandas-first for public review and reproducibility. Spark, GraphFrames, and Colab snippets remain in commented cells so the original distributed-computing setup can be restored for larger datasets.
