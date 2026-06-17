# Energy Data Analysis

Portfolio project developed during the MSc in Business Analytics at NOVA School of Business and Economics.

Advanced Programming for Data Science group project focused on energy consumption, GDP, population, and estimated CO2 emissions across countries using the Our World in Data energy dataset.

The project contains a small Python analysis class (`EnergyData`) and a showcase notebook that demonstrates the main visualizations for China, the United States, and Portugal.

## Repository Contents

- `energy_data_analysis.py`: reusable `EnergyData` class for downloading, enriching, plotting, and forecasting energy data.
- `energy_data_analysis_showcase.ipynb`: portfolio showcase notebook with the analysis narrative and figures.
- `energy_data_analysis.yml`: original Conda environment export used for the coursework.
- `docs/`: Sphinx documentation sources generated during the project.

## Data Source

The analysis downloads the public Our World in Data energy dataset:

<https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.csv>

The downloaded CSV is intentionally not committed. Running the setup cell in the notebook or calling `EnergyData.download_file()` creates a local `downloads/` directory.

## Quick Start

Create the original Conda environment:

```bash
conda env create -f energy_data_analysis.yml
conda activate energy_data_analysis
```

Run the notebook:

```bash
jupyter notebook energy_data_analysis_showcase.ipynb
```

Minimal Python usage:

```python
from energy_data_analysis import EnergyData

energy_data = EnergyData()
energy_data.download_file()
energy_data.enrich()
energy_data.gapminder(year=2016, country="United States")
```

## Authors

- Ana Margarida Pimentel de Morais
- Francesca Fontanesi
- Daniele Gamberoni
- Federico Maria Brignoli

## License

This portfolio reupload is released under the MIT License. See `LICENSE`.
