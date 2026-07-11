#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 00:56:14 2022

@author: federico
"""

# import the necessary libraries

import numpy as np
import pandas as pd
from pathlib import Path
import requests
import matplotlib.pyplot as plot
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from typing import Union
import warnings

warnings.filterwarnings("ignore")


class EnergyData:
    """
    A class that enables the analysis of country/countries mix of energy consumption and corresponding CO2 emissions throughout the years.
    Such analysis can only be done since 1970.

    Attributes
    ---------------
    self.data_df: pd.DataFrame 
        Dataframe containing the data
    self.file_link: string
        Source url
        
    """

    def __init__(self):
        self.data_df = pd.DataFrame()
        self.file_link = "https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.csv"

    def _as_country_list(self, countries: Union[str, list]) -> list:
        if isinstance(countries, str):
            return [countries]
        return list(countries)

    def _validate_countries(self, countries: Union[str, list]) -> list:
        countries = self._as_country_list(countries)
        available_countries = set(self.data_df["country"].unique())
        missing_countries = [country for country in countries if country not in available_countries]
        if missing_countries:
            raise ValueError(
                "All input countries must be present in the dataset. "
                f"Missing: {', '.join(missing_countries)}"
            )
        return countries

    def _year_mask(self, year: int) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(self.data_df["year"]):
            return self.data_df["year"].dt.year == year
        return self.data_df["year"] == year

    def _consumption_columns(self) -> list:
        aggregate_columns = {
            "fossil_fuel_consumption",
            "low_carbon_consumption",
            "renewable_consumption",
            "renewables_consumption",
            "primary_energy_consumption",
        }
        return [
            column
            for column in self.data_df.columns
            if column.endswith("_consumption") and column not in aggregate_columns
        ]

    def _drop_trailing_unavailable_consumption(
        self, df: pd.DataFrame, consumption_columns: list
    ) -> pd.DataFrame:
        """Remove trailing rows where consumption is unavailable and encoded as zero."""
        if df.empty:
            return df

        df = df.sort_values("year").copy()
        has_consumption = df[consumption_columns].fillna(0).sum(axis=1) > 0
        if not has_consumption.any():
            return df.iloc[0:0]

        return df.loc[: has_consumption[has_consumption].index[-1]]

    # Download data file from URL and save it in downloads

    def download_file(self, save_path: str = "downloads"):
        """
        Downloads the data into a "downloads" directory. Checks if the file is already there.
        
        Parameters
        ---------------
        save_path : string
            Name of the directory where file should be downloaded to        
        
        Returns
        ---------------
        None
        """

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Define the name of the file from the URL and its path
        file_name = self.file_link.split("/")[-1]
        file_path = save_path / file_name

        # If the data file already exists, the method will not download it again
        if file_path.exists():
            print("This file already exists!")
        else:
            r = requests.get(self.file_link, stream=True, timeout=30)
            r.raise_for_status()

            # Write the data from the URL in the file
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

        # Read the file into a pandas dataframe
        self.data_df = pd.read_csv(file_path)

        # Consider only years after 1970 included
        self.data_df = self.data_df[self.data_df["year"] >= 1970].reset_index(drop=True)

        # Convert 'year' to datetime object and set it as index
        self.data_df["year"] = pd.to_datetime(self.data_df["year"], format="%Y")
        self.data_df["idx_year"] = self.data_df["year"]
        self.data_df.set_index("idx_year", inplace=True)

    def enrich(self) -> None:
        """
        Enriches the attribute 'data_df' with the column 'emissions', which estimates emissions in tonnes of CO2
        corresponding to the consumption of each energy type.
        """
        # Define emissions in gCO2 per kWh for each energy source
        source_emissions = {
            "biofuel": 1450,
            "coal": 1000,
            "gas": 455,
            "hydro": 90,
            "nuclear": 5.5,
            "oil": 1200,
            "solar": 53,
            "wind": 14,
        }

        def get_emissions(df_row: pd.Series) -> float:
            """
            Calculates the emissions for a given row of attribute 'df'.
            Parameters:
                df_row (pd.Series): The row for which the emissions should be calculated.
            """
            # Create a list with the new emissions values for every source type
            emissions = []
            for energy_type, emission_value in source_emissions.items():
                # Get the CO2 value for the respective type of source
                consumption = df_row.get(energy_type + "_consumption", np.nan)
                # Recalculating the emission value
                source_emission = (consumption * 1e9) * (emission_value / 1e6)
                # adding the new value in emissions list
                if not np.isnan(source_emission):
                    emissions.append(source_emission)
            # Calculate the sum, for every row, of the total emission value
            total_emissions = sum(emissions)
            return total_emissions

        # Calculating the emission value for every row
        self.data_df["emissions"] = self.data_df.apply(get_emissions, axis=1)

    def list_countries(self):
        """
        Print the countries in the dataset 
        
        Returns
        ---------------
        Prints and returns a list with the names of the countries
        """
        list_countries = list(self.data_df["country"].unique())
        print(list_countries)
        return list_countries

    def consumption_chart(
        self, country: str = "Sweden", normalize: bool = False
    ) -> None:
        """
        Plots an area chart with the different "_consumption" columns of the chosen country, in relative and absolute terms
        
        Parameters
        ---------------
        country : string
            The country we choose to plot the information of 
        normalize : bool
            If the values of each year "_consumption" columns are normalized 
        
        Returns
        ---------------
        None
        
        """
        self._validate_countries(country)

        consumption_cols = self._consumption_columns()
        sub_data_df = self.data_df.loc[self.data_df["country"] == country].copy()
        sub_data_df = self._drop_trailing_unavailable_consumption(
            sub_data_df, consumption_cols
        )
        sub_data_df = sub_data_df[consumption_cols]

        if normalize == False:
            sub_data_df.plot(kind="area", figsize=(20, 10))
            return plot.show()

        normalized = sub_data_df.loc[:, sub_data_df.columns.str.endswith("_consumption")]
        normalized = normalized.div(normalized.sum(axis=1), axis=0)
        normalized.plot(
            kind="area",
            figsize=(20, 10),
            title=f"Energy Consumption Mix of {country}",
        )
        return plot.show()

    # Develop a fourth method that may receive a string with a country or a list of country strings.
    # This method should compare the total of the "_consumption" columns for each of the chosen countries and plot it, so a comparison can be made.

    def consumption_countries(self, listCountries, comparison=False):
        """
        Compares the total consumption of energy of each of the chosen countries throughout the years by plotting a line graph
        
        Parameters
        ---------------
        listCountries : list
            A list of counties  
       
        Returns
        ---------------
        None
        
        """
        countries = self._validate_countries(listCountries)
        consumption_cols = self._consumption_columns()

        df_countries = self.data_df[self.data_df["country"].isin(countries)][
            ["country", "year"] + consumption_cols
        ].copy()
        df_countries = self._drop_trailing_unavailable_consumption(
            df_countries, consumption_cols
        )
        df_countries["sum"] = df_countries[consumption_cols].sum(axis=1)

        df_countries_pivot = df_countries.pivot_table("sum", "year", "country")
        df_countries_pivot.replace({0: np.nan}, inplace=True)

        if not comparison:
            df_countries_pivot.plot()
            plot.show()

        if comparison:
            return df_countries_pivot

    def consumpion_countries(self, listCountries, comparison=False):
        """Backward-compatible alias for the original misspelled method call."""
        return self.consumption_countries(listCountries, comparison)

    # Develop a fifth method that may receive a string with a country or a list of country strings.
    # This method should compare the "gdp" column of each country over the years.

    def gdp_countries_chart(self, listCountries: list) -> None:
        """
        Compares the GDP of each of the chosen countries throughout the years by plotting a line graph
        
        Parameters
        ---------------
        listCountries : list
            A list of counties  
       
        Returns
        ---------------
        None
        
        """
        countries = self._validate_countries(listCountries)
        df_countries = self.data_df[self.data_df["country"].isin(countries)]
        df_countries_pivot = df_countries.pivot_table("gdp", "year", "country")
        df_countries_pivot.plot(figsize=(20, 10), title="GDP per year of Countries")
        plot.show()

    def gapminder(self, year: int, country=None):
        """
        Plots a scatter of GDP against Energy Consumption according to the year selected
        The dimension of the marks is proportional to the population of the country
        
        Parameters
        ---------------
        year : int
            The year we are analysing 
        
        country : str
            The country that I want to highlight in the gapminder plot
        
        Returns
        ---------------
        None
        """
        if not isinstance(year, int):
            raise TypeError("TypeError: The year must be an int value")

        # Retrieve data from specified year
        gapminder_year = self.data_df[self._year_mask(year)].copy()
        if gapminder_year.empty:
            raise ValueError(f"No data is available for year {year}.")

        # remove aggregate data/not country-level data
        gapminder_year = gapminder_year[
            gapminder_year["iso_code"] != "OWID_WRL"
        ].dropna(subset=["iso_code"])

        # remove aggregate columns ['fossil_fuel_consumption', 'low_carbon_consumption', 'renewable_consumption', 'primary_energy_consumption']

        consumptions = self._consumption_columns()
        gapminder_year["total"] = gapminder_year[consumptions].sum(axis=1)

        sns.scatterplot(
            data=gapminder_year, x="gdp", y="total", size="population", alpha=0.5
        )
        plot.title("GDP vs Total Energy Consumption")
        plot.xscale("log")
        if country is not None:
            self._validate_countries(country)
            edfpt = gapminder_year[gapminder_year["country"] == country].copy()
            edfpt["total"] = edfpt[consumptions].sum(axis=1)
            sns.scatterplot(
                data=edfpt, x="gdp", y="total", size="population", color="orange"
            )

        plot.show()
        plot.clf()

    def plot_emissions(self, listCountries: Union[str, list]) -> None:
        """
        Creates a plot that compares the total energy consumption and emissions for given countries.
        The total energy consumption is represented with a continuous line and bound to the left y-axis whilst the
        emissions are represented with a dashed line and bound to the right y-axis.

        Parameters:
            countries (str/list): The country or countries to plot the energy consumption for.

        Raises:
            ValueError: If one of the countries passed in 'countries' is not present in the dataset.
        """

        countries = self._validate_countries(listCountries)
        if "emissions" not in self.data_df.columns:
            self.enrich()

        df_countries_cons = self.consumption_countries(countries, True)

        df_countries_em = self.data_df[self.data_df["country"].isin(countries)][
            ["country", "emissions", "year"]
        ]
        df_countries_em = df_countries_em.pivot_table("emissions", "year", "country")
        df_countries_em.replace({0: np.nan}, inplace=True)
        df_countries_em = df_countries_em.reindex(df_countries_cons.index)

        ax = plot.subplots()[1]
        ax2 = ax.twinx()

        df_countries_cons.plot(ax=ax)
        df_countries_em.plot(linestyle="--", ax=ax2)

        ax.set_title("Energy Consumption & Emissions per Country")
        ax.set_xlabel("Year")
        ax.set_ylabel("Consumption in TWh")
        ax2.set_ylabel("Emissions in Tonnes of CO2")
        ax.legend(title="Consumption", loc="upper left", bbox_to_anchor=(1.1, 1))
        ax2.legend(title="Emissions", loc="lower left", bbox_to_anchor=(1.1, 0))

        plot.show()

    def prediction(self, country: str, periods: int) -> None:

        """Plots the predicted consumption and the predicted emissions for the country over the periods selected
    
        Parameters
        ---------------
        country: String 
            The country to plot the predictions for.
        periods: int
            The number of periods (years) to make predictions for.
        """

        if not isinstance(periods, int):
            raise TypeError("periods is not of type int")

        if periods < 1:
            raise ValueError("Periods must be bigger or equal to 1")

        self._validate_countries(country)
        if "emissions" not in self.data_df.columns:
            self.enrich()

        # Filter data relative to input country
        country_df = self.data_df[self.data_df["country"] == country].copy()

        # Consider only consumptions that are not of multiple energy sources
        columns = self._consumption_columns()

        country_df = self._drop_trailing_unavailable_consumption(country_df, columns)

        # Sum of consumptions
        country_df["tot_consumption"] = country_df[columns].sum(axis=1)

        country_df = country_df[["year", "tot_consumption", "emissions"]]

        # Drop rows containing NaN
        country_df = country_df.dropna()

        # Consuption model with ARIMA
        model_consumption = ARIMA(country_df["tot_consumption"], order=(5, 1, 2))
        model_fit_consumption = model_consumption.fit()
        prediction_consumption = model_fit_consumption.predict(
            len(country_df["tot_consumption"]),
            len(country_df["tot_consumption"]) + periods - 1,
        )
        consumption_df = pd.Series.to_frame(prediction_consumption)

        fig, axes = plot.subplots(nrows=1, ncols=2, figsize=(12, 4))

        # Consumption plot
        axes[0].plot(
            country_df.index,
            country_df["tot_consumption"],
            color="blue",
            label="Actual",
        )
        axes[0].plot(
            consumption_df.index,
            consumption_df["predicted_mean"],
            c="red",
            label="Predicted",
        )
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("TWh")
        axes[0].set_title(
            f"Prediction of total consumption in {country} for {periods} years"
        )

        # Emissions model with ARIMA
        model_emissions = ARIMA(country_df["emissions"], order=(5, 1, 2))
        model_fit_emissions = model_emissions.fit()
        prediction_emissions = model_fit_emissions.predict(
            len(country_df["emissions"]), len(country_df["emissions"]) + periods - 1
        )
        emissions_df = pd.Series.to_frame(prediction_emissions)

        # Emissions plot
        axes[1].plot(country_df.index, country_df["emissions"], label="Actual")
        axes[1].plot(
            emissions_df.index,
            emissions_df["predicted_mean"],
            color="red",
            label="Predicted",
        )
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("CO2 Tonnes")
        axes[1].set_title(
            f"Prediction of total emissions in {country} for {periods} years"
        )

        plot.legend(loc="lower right", bbox_to_anchor=(1, -0.3), ncol=2)

        plot.show()

    def scatter_emissions_consumption(self, year: int = None):
        """Scatter plot between emissions (x) and consumption (y) for all countries. The size of the dots represents the population
    
        Parameters
        ---------------
            year: optional integer year to filter the plot.
      
        For a given year, creates a scatter plot where the x-axis is the emissions, the y-axis is the total energy
        consumption, and the size of each dot is the population of the corresponding country.
        """

        if "emissions" not in self.data_df.columns:
            self.enrich()

        scatter_df = self.data_df.copy()
        if year is not None:
            if not isinstance(year, int):
                raise TypeError("TypeError: The year must be an int value")
            scatter_df = scatter_df[self._year_mask(year)].copy()

        # Remove values that refer to the world and NaNs
        scatter_df = scatter_df[scatter_df["iso_code"] != "OWID_WRL"].dropna(
            subset=["iso_code"]
        )

        columns = self._consumption_columns()
        scatter_df["tot_consumption"] = scatter_df[columns].sum(axis=1)
        scatter_df = scatter_df.dropna(subset=["emissions", "tot_consumption", "population"])
        scatter_df = scatter_df[
            (scatter_df["emissions"] > 0) & (scatter_df["tot_consumption"] > 0)
        ]

        sns.scatterplot(
            data=scatter_df,
            x="emissions",
            y="tot_consumption",
            size="population",
            color="green",
        )
        plot.title("Emissions vs. Consumption vs. Population")
        plot.xscale("log")
        plot.yscale("log")
        plot.xlabel("Emissions (in tonnes of CO2)")
        plot.ylabel("Energy Consumption in TWh")

        # Legend
        plot.legend(title="Population\n(billion)")

        plot.show()
