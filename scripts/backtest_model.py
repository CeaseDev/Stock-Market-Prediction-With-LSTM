import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the trained model
model_path = '/mnt/e/Stock_prediction_major/models/lstm_stock_model.h5'
model = tf.keras.models.load_model(model_path)

# Load the test data for a specific stock

def load_test_data_for_stock(ticker, data_path, holdout_days=10):
    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    sequence_length = 60

    file_path = os.path.join(data_path, f"{ticker}_preprocessed.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None, None

    df = pd.read_csv(file_path)
    df.dropna(inplace=True)

    # Select only numeric columns, excluding 'Date' and other non-numeric columns
    df = df.select_dtypes(include=[np.number])
    df[df.columns] = scaler.fit_transform(df[df.columns])
    df['Close'] = target_scaler.fit_transform(df[['Close']])
    data = df.values

    # Prepare testing data
    test_data = data[-(holdout_days + sequence_length):]
    X_test = []
    y_test = []

    for i in range(sequence_length, len(test_data)):
        X_test.append(test_data[i-sequence_length:i])
        y_test.append(test_data[i, 3])  # Assuming 'Close' is at index 3

    X_test = np.array(X_test, dtype='float32')
    y_test = np.array(y_test, dtype='float32')

    return X_test, y_test, target_scaler

if __name__ == "__main__":
    # Specify paths and tickers
    data_path = '/mnt/e/Stock_prediction_major/data/stocks'
    tickers = [
    "RELIANCE", "TCS", "INFY", "ICICIBANK", "ASIANPAINT", 
    "KOTAKBANK", "LT", "HINDUNILVR", "BAJFINANCE", "ITC",
    "BHARTIARTL", "SBIN", "MARUTI", "HDFCBANK",
    "ADANIGREEN", "WIPRO", "TECHM", "HCLTECH", "POWERGRID" ]  # Add more tickers as needed

    for ticker in tickers:
        print(f"Backtesting for {ticker}...")
        X_test, y_test, target_scaler = load_test_data_for_stock(ticker, data_path)
        
        if X_test is None:
            continue
        
        # Make predictions
        predictions = model.predict(X_test)

        # Inverse scale the predictions and actual values
        y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        predictions = target_scaler.inverse_transform(predictions).flatten()

        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"{ticker} - Mean Absolute Error (MAE): {mae}")
        print(f"{ticker} - Root Mean Squared Error (RMSE): {rmse}")

        # Plot actual vs. predicted values
        plt.figure(figsize=(14, 7))
        plt.plot(y_test, label='Actual Prices', color='blue')
        plt.plot(predictions, label='Predicted Prices', color='red')
        plt.title(f'Actual vs. Predicted Stock Prices for {ticker}')
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

 # Add more tickers as needed
