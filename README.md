# Bitcoin Price Prediction Dashboard

This project provides a real-time forecasting system for Bitcoin (BTC-USD). It includes a stacked LSTM neural network for predicting Open, High, Low, and Close (OHLC) prices and an interactive Streamlit dashboard for visualizing current and future trends.

## Features

- Multi-Feature Prediction: Simultaneously forecasts Open, High, Low, and Close prices.
- Real-Time Data: Fetches the latest market information automatically from Yahoo Finance.
- Recursive Forecasting: Predicts future price movements for up to 30 days ahead by using model-generated values as inputs for subsequent steps.
- Interactive Dashboard: Built with Streamlit and Plotly for high-quality, interactive financial charting.

## Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (Recommended for training, though CPU is supported)

## Dependencies

The following Python libraries are required:
- tensorflow
- streamlit
- yfinance
- pandas
- numpy
- matplotlib
- plotly
- scikit-learn
- joblib

## Setup Instructions

1. Create a virtual environment:
   ```powershell
   python -m venv venv_win
   ```

2. Activate the virtual environment:
   ```powershell
   .\venv_win\Scripts\activate
   ```

3. Install the necessary requirements:
   ```powershell
   pip install tensorflow streamlit yfinance pandas numpy matplotlib plotly scikit-learn joblib
   ```

## Usage

### 1. Training the Model
To train the LSTM model on 5 years of historical Bitcoin data and generate the necessary scaling artifacts:
```powershell
python btc_lstm.py
```
This script will produce:
- `btc_lstm_model.keras`: The trained neural network.
- `scaler.joblib`: The persistence file for data normalization.
- `results.txt`: Performance metrics (RMSE, MAE, R2).

### 2. Launching the Dashboard
To start the interactive web application:
```powershell
streamlit run app.py
```
The dashboard will be available at `http://localhost:8501`.

## Project Structure

- `app.py`: The Streamlit web application and forecasting engine.
- `btc_lstm.py`: The training logic and model architecture.
- `btc_lstm_model.keras`: Trained model file.
- `scaler.joblib`: Saved MinMaxScaler parameters.
- `results.txt`: Detailed accuracy metrics for all predicted features.
- `btc_prediction_plot.png`: Static visualization of the latest test set performance.

## Disclaimer
This tool is for educational and research purposes only. Financial markets involve significant risk, and predictions made by this model should not be used as professional investment advice.
