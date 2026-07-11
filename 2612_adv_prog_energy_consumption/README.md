# Energy Data Analysis

Project developed for the course Advanced Programming for Data Science during the MSc in Business Analytics at NOVA School of Business and Economics.

Analysis focused on energy consumption, GDP, population, and estimated CO2 emissions across countries using the Our World in Data energy dataset.

The project contains a small Python analysis class (`EnergyData`) and a showcase notebook that demonstrates the main visualizations for China, the United States, and Portugal.

## Repository Contents

- `energy_data_analysis.py`: reusable `EnergyData` class for downloading, enriching, plotting, and forecasting energy data.
- `energy_data_analysis_showcase.ipynb`: portfolio showcase notebook with the analysis narrative and figures.
- `environment.yml`: concise Conda environment for the portfolio workflow.
- `energy_data_analysis.yml`: original Conda environment export used for the coursework.
- `docs/`: Sphinx documentation sources generated during the project.

## Data Source

The analysis downloads the public Our World in Data energy dataset:

<https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.csv>

The downloaded CSV is intentionally not committed. Running the setup cell in the notebook or calling `EnergyData.download_file()` creates a local `downloads/` directory.

## Authors

- Ana Margarida Pimentel de Morais
- Francesca Fontanesi
- Daniele Gamberoni
- Federico Maria Brignoli


