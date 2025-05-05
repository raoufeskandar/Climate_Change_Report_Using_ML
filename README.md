# Climate_Change_Report_Using_ML

# Climate Change Impact Dashboard

This project provides a data-driven analysis of climate change impacts across the globe using a real-world dataset. It includes data preprocessing, machine learning-based clustering and classification, and a fully interactive dashboard built with Streamlit.

---

## Dataset Overview

The main dataset used in this project is `realistic_climate_change_impacts.csv`. It contains records for different countries and includes the following columns:

- `Country`: Name of the country where the event occurred
- `ExtremeWeatherEvent`: Description of the climate-related event
- `CO2Level_ppm`: CO₂ levels measured in parts per million
- `TemperatureAnomaly_C`: Change in temperature from historical baseline
- `PopulationAffected`: Number of people affected
- `EconomicImpact_USD`: Estimated financial cost due to the event

Additionally, a shapefile (`ne_110m_admin_0_countries`) is used for the geographic visualization on the world map.

---

## Project Structure
realistic_climate_change_impacts/
├── .venv/                  # Virtual environment
├── app/                    # Streamlit application
│   └── streamlit_app.py    # Interactive dashboard
├── data/                   # Data files
│   ├── ne_110m_admin_0_countries.*  # Shapefiles for world map
│   └── realistic_climate_change_impacts.csv  # Main dataset
├── models/                 # Trained machine learning models
│   ├── classification_model.pkl
│   ├── clustering_model.pkl
│   └── scaler.pkl
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_cleaning_and_eda.ipynb  # Data cleaning and EDA
│   ├── wordcloud_event_descriptions.png  # Generated word cloud
├── pyvenv.cfg              # Virtual environment configuration
├── requirements.txt        # Project dependencies
├── README.md               # This file


## Installation
    Python 3.13.2

## Setup Steps
# 1- Create a Virtual Environment:
    python -m venv .venv
    source .venv/Scripts/activate  # Windows
    Or: source .venv/bin/activate  # Mac/Linux

# 2- Install Dependencies:
    pip install -r requirements.txt

# Running the Jupyter Notebook
    jupyter notebook



## Code Overview
The 01_data_cleaning_and_eda.ipynb notebook is the core of the analysis. Below is a breakdown of its sections:

    1. Library Imports
        Imports standard libraries (pandas, numpy, os, random) and specialized ones (geopandas, seaborn, matplotlib, sklearn, wordcloud, transformers, ipywidgets, IPython).
        Note: geopandas is imported multiple times (duplicated), which is redundant but harmless.

    2. Data Loading and Initial Exploration
        Loads realistic_climate_change_impacts.csv into a pandas DataFrame.
        Cleans column names (e.g., converts to lowercase, replaces spaces with underscores).
        Displays the first 5 rows (df.head()), data info (df.info()), and null value counts.
        Analyzes extremeweatherevent distribution, revealing 1,384 missing values.

    3. Data Cleaning
        Converts economicimpact_usd from string to float (removes commas).
        Converts date to datetime format.
        Verifies data types and null counts post-cleaning.

    4. Handling Missing Data
        Fills missing extremeweatherevent values with "Unknown".
        Confirms the fix with value counts.

    5. NLP Simulation with Word Cloud
        Generates fake event descriptions by combining random words (e.g., "hurricane", "flood") with extremeweatherevent values.
        Creates a word cloud using WordCloud with a plasma colormap, saved as wordcloud_event_descriptions.png.

    6. AI Analysis with Transformers
        Initializes a distilgpt2 text generation model.
        Creates interactive widgets:
        Combobox for country selection.
        Dropdown for calculation method ("per year", "per day", "square foot").
        Buttons to generate AI analysis and calculate impacts.
        Computes average CO₂ levels per country and generates text using a prompt (e.g., "The CO2 consumption of {country} is {avg_co2:.2f} ppm...").
        Calculates economic impact based on the selected method (e.g., divides by 365 for "per day").

    7. Environment Check
        Prints the Python executable path (sys.executable) to verify the environment.

## streamlit_app.py
    This is the Streamlit app to visualize everything. It includes:
    - Data Filters: Sidebar sliders to adjust the range of CO₂, temperature, economic impact, and population
    - Navigation Pages:
        - Raw and Cleaned Data
        - Top countries by event count
        - Correlation Heatmap
        - KMeans Clustering
        - Classification Predictions
        - Feature Importance (from classifier)
        - World Map (GeoPandas + Plotly)
        - Word Cloud of events
        - Model Integration:
        - Reuses pre-trained clustering and classification models
        - Performs real-time inference on filtered data

# Run the Streamlit Dashboard
    1. From project root:
        streamlit run app/streamlit_app.py

    2. Open in your browser:
         http://localhost:8501 or http://localhost:8502/
