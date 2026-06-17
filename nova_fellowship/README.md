# Portugal Oral Health Voucher Program

Exploratory analysis and forecasting project prepared as part of a Data Science fellowship portfolio at the NOVA SBE Data Science Knowledge Centre, now DOT Centre.

The notebook analyzes public monthly data on Portugal's oral health voucher program (`cheques-dentista`), a National Health Service initiative that issues dental-care vouchers to eligible groups including children and young people, pregnant SNS users, older adults receiving social support, people living with HIV/AIDS, and users covered by oral-cancer early-intervention pathways.

## Contents

- `analysis_saúde_oral.ipynb`: end-to-end notebook for data validation, descriptive analysis, monthly dynamics, time-series diagnostics, and predictive modeling.
- `LICENSE`: project license.

## Analysis Covered

1. Data validation and cleaning for monthly program data from January 2017 to July 2022.
2. Descriptive summaries by entity, target population, and intervention scope.
3. Monthly trend and seasonality analysis for issued, used, and treated vouchers.
4. Time-series diagnostics using additive seasonal decomposition.
5. Predictive modeling with seasonal baselines, regression, decision tree, and random forest models.

## Running the Notebook

Install the Python dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter
```

The notebook expects the source CSV at:

```text
data/saude-oral.csv
```

Then open and run:

```bash
jupyter notebook analysis_saúde_oral.ipynb
```

## Notes

- Keep raw or sensitive data out of Git unless it is explicitly intended for public release.
- If the data file is not committed, include a source link or download instructions before publishing.
