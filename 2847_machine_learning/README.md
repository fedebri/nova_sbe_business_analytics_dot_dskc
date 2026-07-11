# H&M Personalized Fashion Recommendations

Machine learning coursework project for the H&M Personalized Fashion Recommendations Kaggle competition.

The project explores customer, article, and transaction data, then builds baseline, collaborative-filtering, and content-based recommendation approaches.

## Project Structure

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

Small local development samples may live under `data/sample`, but all data paths are ignored by git by default to avoid accidental leakage.

## Running

Create the environment:

```bash
conda env create -f environment_ml2847.yml
conda activate ml2847
```

Run the notebooks in this order:

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
