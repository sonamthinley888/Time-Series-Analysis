# Time-Series-Analysis
# Time Series Analysis on Jetrail Traffic dataset
# Author

Sonam Thinley

# Overview

This project involves time series analysis and forecasting using various statistical and machine learning techniques. The dataset consists of hourly count data collected over a period, and we aim to preprocess, visualize, and apply models such as ARIMA and SARIMA to predict future values.

# Dataset

The dataset is stored in Train.csv and contains three columns:

ID: Unique identifier

Datetime: Timestamp

Count: The observed value

# Project Steps

1. Data Import and Exploration

Load the dataset using Pandas.

Display the first and last few rows to understand the structure.

Check for missing values and data types.

2. Data Preprocessing

Convert the Datetime column to a proper datetime format.

Set Datetime as the index and resample the data to a daily mean.

Drop unnecessary columns (e.g., ID).

3. Train-Test Split

Split the dataset into training (80%) and testing (20%) subsets.

4. Data Visualization

Plot the time series data to observe trends and seasonality.

5. Stationarity Check

Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity.

If the data is non-stationary, apply seasonal differencing.

6. Autocorrelation and Partial Autocorrelation

Use ACF and PACF plots to determine the optimal parameters for ARIMA and SARIMA models.

7. Time Series Models

# ARIMA Model

Fit an ARIMA model with determined parameters.

Forecast future values and compare with the test set.

Compute RMSE to evaluate performance.

# SARIMA Model

Extend ARIMA to Seasonal ARIMA (SARIMA) by incorporating seasonality.

Fit the model and visualize the predictions.

Compute RMSE for performance comparison.

# Dependencies

Ensure the following Python libraries are installed before running the notebook:

pip install numpy pandas matplotlib statsmodels

# Running the Code

Clone the repository:

git clone <repo_url>
cd <repo_name>

Run the Jupyter Notebook or Python script to execute the analysis.

# Results

The best-performing model is determined based on RMSE.

Forecasting results are visualized to assess the accuracy.

Future Improvements

Experiment with additional models such as LSTMs for deep learning-based forecasting.

Fine-tune hyperparameters for better model performance.

