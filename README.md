# Dimensionality Reduction Ireland

A machine learning project for rainfall prediction across Irish counties using hybrid LSTM-Prophet models with dimensionality reduction techniques.

## Overview

This project implements advanced time series forecasting models to predict rainfall amounts across different counties in Ireland. The approach combines:

- **LSTM (Long Short-Term Memory)** neural networks for capturing long-term temporal dependencies
- **Prophet** time series forecasting for trend and seasonality modeling
- **Dimensionality reduction** techniques for feature optimization
- **Multi-county analysis** covering diverse geographical regions

## Features

- Hybrid LSTM-Prophet model implementation
- Comprehensive weather feature analysis (17 meteorological variables)
- Multi-county rainfall prediction across 6 Irish regions
- Advanced feature engineering and preprocessing
- Performance evaluation and visualization

## Dataset

The project includes weather data for the following Irish counties:

- **Carlow Oak Park** - `county_carlowoakpark.parquet`
- **Cavan** - `county_cavan.parquet`
- **Clare (Shannon Airport)** - `county_clare.parquet`
- **Donegal** - `county_donnegal.parquet`
- **Galway** - `county_galway.parquet`
- **Westmeath** - `county_westmeath.parquet`

### Meteorological Variables

The datasets contain 17 features including:
- `maxtp` - Maximum temperature
- `mintp` - Minimum temperature
- `gmin` - Grass minimum temperature
- `rain` - Rainfall amount (target variable)
- `cbl` - Cloud base level
- `wdsp` - Wind speed
- `hm` - Humidity measures
- `soil` - Soil temperature
- `pe` - Potential evapotranspiration
- `evap` - Evaporation
- `smd_wd`, `smd_md`, `smd_pd` - Soil moisture deficit measures
- `glorad` - Global radiation

## Repository Structure

```
dimensionalityreductionireland/
├── datasets/
│   ├── county_carlowoakpark.parquet
│   ├── county_cavan.parquet
│   ├── county_clare.parquet
│   ├── county_donnegal.parquet
│   ├── county_galway.parquet
│   └── county_westmeath.parquet
├── scripts/
│   ├── LSTMProphet-AllFeatures-rainfallamt-carlowoakpark.ipynb
│   ├── LSTMProphet-AllFeatures-rainfallamt-cavan.ipynb
│   ├── LSTMProphet-AllFeatures-rainfallamt-clareshannonairport.ipynb
│   ├── LSTMProphet-AllFeatures-rainfallamt-donnegal.ipynb
│   ├── LSTMProphet-AllFeatures-rainfallamt-galway.ipynb
│   └── LSTMProphet-AllFeatures-rainfallamt-westmeath.ipynb
└── README.md
```

## Requirements

The project requires the following Python packages:

```
pandas
numpy
scikit-learn
tensorflow
keras
prophet
matplotlib
seaborn
pyarrow (for parquet files)
jupyter
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/menatallahMohamed/dimensionalityreductionireland.git
cd dimensionalityreductionireland
```

2. Install required dependencies:
```bash
pip install pandas numpy scikit-learn tensorflow prophet matplotlib seaborn pyarrow jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## Usage

1. **Data Exploration**: Each notebook begins with data loading and exploratory analysis
2. **Preprocessing**: Features are scaled and prepared for model training
3. **Model Training**: Hybrid LSTM-Prophet models are trained on historical data
4. **Prediction**: Models generate rainfall forecasts for each county
5. **Evaluation**: Performance metrics and visualizations assess model accuracy

### Running a County Analysis

Open any of the county-specific notebooks in the `scripts/` directory:

```python
# Example for Galway county
import pandas as pd
county_galway = pd.read_parquet("../datasets/county_galway.parquet")
# Follow the notebook steps for complete analysis
```

## Model Architecture

The hybrid approach combines:

1. **LSTM Component**: Captures temporal patterns and long-term dependencies
2. **Prophet Component**: Models trend, seasonality, and holiday effects
3. **Feature Engineering**: Dimensionality reduction optimizes input variables
4. **Ensemble Method**: Combines predictions for improved accuracy

## Results

Each notebook provides:
- Model performance metrics
- Prediction visualizations
- Feature importance analysis
- Comparative analysis across counties

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

Menatallah Mohamed

---

*This project demonstrates the application of advanced machine learning techniques for weather prediction in Ireland, combining deep learning with traditional time series forecasting methods.*
