# Time Series Forecasting Project

# Overview

This project involves time series analysis and forecasting using various statistical and deep learning techniques. The dataset consists of hourly count data collected over a period, and we aim to preprocess, visualize, and apply models such as ARIMA, SARIMA, and LSTMs to predict future values.
# Author

Sonam Thinley
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

Split the dataset into training (70%) and testing (30%) subsets.

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

# LSTM Model

Implement an LSTM-based neural network for time series forecasting.

Preprocess data to create sequences for LSTM training.

Define an LSTM model using PyTorch:

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

Train the LSTM model on the time series data and evaluate RMSE.

Visualize the predictions against actual values.

# Dependencies

Ensure the following Python libraries are installed before running the notebook:

pip install numpy pandas matplotlib statsmodels torch

# Running the Code

Clone the repository:

git clone <repo_url>
cd <repo_name>

Run the Jupyter Notebook or Python script to execute the analysis.

# Results

The best-performing model is determined based on RMSE.

Forecasting results are visualized to assess the accuracy.

# Future Improvements

Experiment with additional models such as GRUs and Transformers for better forecasting.

Fine-tune hyperparameters for improved model performance.


