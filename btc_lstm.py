import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
import os
import joblib

# --- Configuration ---
SYMBOL = 'BTC-USD'
FEATURES = ['Open', 'High', 'Low', 'Close']
LOOKBACK_STEPS = 60
TRAIN_SPLIT = 0.8
EPOCHS = 20
BATCH_SIZE = 32

def main():
    print(f"--- Bitcoin OHLC Price Prediction System ---")
    
    # 1. Load Bitcoin historical data
    print(f"Step 1: Fetching 5 years of historical data for {SYMBOL}...")
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=5*365)
    df = yf.download(SYMBOL, start=start_date, end=end_date)
    
    if df.empty:
        print("Error: Could not fetch data. Please check your internet connection or the symbol.")
        return

    # Use Open, High, Low, and Close prices
    data = df[FEATURES].values
    
    # 2. Normalize data using MinMaxScaler
    print(f"Step 2: Normalizing data for features: {FEATURES}...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 3. Create sequences of 60 timesteps (X) to predict next values (y)
    print(f"Step 3: Creating sequences of {LOOKBACK_STEPS} timesteps...")
    X, y = [], []
    for i in range(LOOKBACK_STEPS, len(scaled_data)):
        X.append(scaled_data[i-LOOKBACK_STEPS:i, :])
        y.append(scaled_data[i, :])
    
    X, y = np.array(X), np.array(y)
    
    # 4. Split dataset into 80% training and 20% testing without shuffling
    print(f"Step 4: Splitting data (Train: {int(TRAIN_SPLIT*100)}%, Test: {int((1-TRAIN_SPLIT)*100)}%)...")
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 5. Model architecture
    print("Step 5: Building multi-output LSTM model...")
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=64),
        Dropout(0.2),
        Dense(units=len(FEATURES)) # Predicting 4 values
    ])
    
    # 6. Training
    print("Step 6: Training model...")
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Check GPU availability
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"CUDA Cores detected. Using GPU: {gpus[0].name}")
    else:
        print("No GPU detected. Falling back to CPU.")
        
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # 7. Predict on test data
    print("Step 7: Predicting on test data...")
    predictions = model.predict(X_test)
    
    # Inverse transform predictions
    print("Step 8: Inverse transforming results...")
    # predictions and y_test are both (samples, 4)
    predictions_unscaled = scaler.inverse_transform(predictions)
    y_test_unscaled = scaler.inverse_transform(y_test)
    
    # 8. Plot actual vs predicted prices for all features
    print("Step 9: Plotting results...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'{SYMBOL} OHLC Price Prediction - LSTM', fontsize=16)
    
    for i, feature in enumerate(FEATURES):
        ax = axes[i // 2, i % 2]
        ax.plot(y_test_unscaled[:, i], color='blue', label=f'Actual {feature}')
        ax.plot(predictions_unscaled[:, i], color='red', linestyle='--', label=f'Predicted {feature}')
        ax.set_title(f'{feature} Price')
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('btc_prediction_plot.png')
    print("Plot saved as 'btc_prediction_plot.png'")
    
    # 9. Calculate Metrics for each feature
    print(f"Step 10: Metrics Calculated:")
    
    feature_metrics = {}
    for i, feature in enumerate(FEATURES):
        actual = y_test_unscaled[:, i]
        pred = predictions_unscaled[:, i]
        
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        
        feature_metrics[feature] = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
        print(f"  {feature}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%, R2={r2:.4f}")

    # Save results to file
    with open('results.txt', 'w') as f:
        f.write(f"Bitcoin OHLC Price Prediction Results\n")
        f.write(f"====================================\n")
        f.write(f"Symbol: {SYMBOL}\n")
        f.write(f"Features predicted: {', '.join(FEATURES)}\n")
        f.write(f"Time Range: Last 5 Years\n")
        f.write(f"Model: stacked LSTM (64, 64 units)\n")
        f.write(f"Training Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"------------------------------------\n")
        
        for feature, metrics in feature_metrics.items():
            f.write(f"{feature} Metrics:\n")
            f.write(f"  - RMSE: {metrics['RMSE']:.2f}\n")
            f.write(f"  - MAE: {metrics['MAE']:.2f}\n")
            f.write(f"  - MAPE: {metrics['MAPE']:.2f}%\n")
            f.write(f"  - R2 Score: {metrics['R2']:.4f}\n")
            f.write(f"------------------------------------\n")
            
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
    print("Results saved to 'results.txt'")

    # 10. Save the model and scaler
    model.save('btc_lstm_model.keras')
    print("Model saved as 'btc_lstm_model.keras'")
    joblib.dump(scaler, 'scaler.joblib')
    print("Scaler saved as 'scaler.joblib'")

if __name__ == "__main__":
    main()
