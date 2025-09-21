# Dimensionality Reduction Ireland - Rainfall Prediction Analysis

This repository contains machine learning analysis for rainfall prediction across various counties in Ireland using LSTM neural networks, Prophet time series forecasting, and dimensionality reduction techniques.

## Project Overview

The project focuses on predicting rainfall amounts in Irish counties using historical weather data. It implements and compares different machine learning approaches:

- **LSTM (Long Short-Term Memory)** neural networks for time series forecasting
- **Prophet** time series forecasting model
- **Dimensionality Reduction** techniques including:
  - Standard PCA (Principal Component Analysis)
  - Kernel PCA for non-linear dimensionality reduction

## Datasets

The repository includes weather data for 6 Irish counties stored in Parquet format:

- `county_carlowoakpark.parquet`
- `county_cavan.parquet`
- `county_clare.parquet`
- `county_donnegal.parquet`
- `county_galway.parquet`
- `county_westmeath.parquet`

### Data Features

Each dataset contains comprehensive weather measurements including:
- **date**: Timestamp of the observation
- **maxtp**: Maximum temperature
- **mintp**: Minimum temperature
- **gmin**: Ground minimum temperature
- **rain**: Rainfall amount (target variable)
- **cbl**: Cloud base level
- **wdsp**: Wind speed
- **hm**: Humidity measurements
- **ddhm**: Degree days humidity
- **hg**: Height gauge
- **soil**: Soil temperature
- **pe**: Potential evaporation
- **evap**: Evaporation
- **smd_wd**: Soil moisture deficit (well drained)
- **smd_md**: Soil moisture deficit (moderately drained)
- **smd_pd**: Soil moisture deficit (poorly drained)
- **glorad**: Global radiation

## Analysis Notebooks

The repository contains Jupyter notebooks for each county implementing the machine learning pipeline:

- `LSTMProphet-AllFeatures-rainfallamt-carlowoakpark.ipynb`
- `LSTMProphet-AllFeatures-rainfallamt-cavan.ipynb`
- `LSTMProphet-AllFeatures-rainfallamt-clareshannonairport.ipynb`
- `LSTMProphet-AllFeatures-rainfallamt-donnegal.ipynb`
- `LSTMProphet-AllFeatures-rainfallamt-galway.ipynb`
- `LSTMProphet-AllFeatures-rainfallamt-westmeath.ipynb`

Each notebook implements:
1. **Data Loading and Exploration**: Initial data analysis and visualization
2. **LSTM with All Features**: Time series forecasting using all available features
3. **LSTM with Kernel PCA**: Non-linear dimensionality reduction followed by LSTM
4. **LSTM with Standard PCA**: Linear dimensionality reduction followed by LSTM
5. **Model Comparison**: Performance evaluation and comparison

## Requirements

To run the analysis notebooks, you'll need the following Python packages:

```
pandas
numpy
seaborn
matplotlib
scikit-learn
tensorflow/keras (for LSTM)
prophet
pyarrow (for reading Parquet files)
jupyter
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/menatallahMohamed/dimensionalityreductionireland.git
cd dimensionalityreductionireland
```

2. Install dependencies:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow prophet pyarrow jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## Usage

1. Navigate to the `scripts/` directory
2. Open any of the county-specific notebooks
3. Run the cells sequentially to:
   - Load and explore the weather data
   - Train LSTM models with different feature sets
   - Apply dimensionality reduction techniques
   - Compare model performance

## Methodology

### LSTM Approach
- Uses all weather features as input variables
- Implements lagged features for time series prediction
- Trains deep learning models to predict rainfall amounts

### Dimensionality Reduction
- **Standard PCA**: Linear transformation to reduce feature dimensionality
- **Kernel PCA**: Non-linear dimensionality reduction using RBF kernels
- Compares prediction performance with and without dimensionality reduction

### Model Evaluation
- Uses multiple metrics to evaluate model performance
- Compares different approaches for each county
- Analyzes the impact of dimensionality reduction on prediction accuracy

## Results

Each notebook provides:
- Model performance metrics
- Visualizations of predictions vs actual values
- Comparison of different approaches
- Analysis of feature importance and dimensionality reduction effectiveness

## Contributing

Feel free to contribute to this project by:
- Adding new counties or regions
- Implementing additional dimensionality reduction techniques
- Exploring other machine learning models
- Improving data preprocessing and feature engineering

## License

This project is open source. Please check with the repository owner for specific licensing terms.

## Contact

For questions or collaboration, please reach out through the GitHub repository.