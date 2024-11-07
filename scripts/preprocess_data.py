import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import ta  # Technical Analysis library

# Function to read and preprocess stock data and sentiment data
def preprocess_stock_data(tickers, stock_data_path, sequence_length=60):
    data = []
    scalers = {}

    for ticker in tickers:
        stock_file = os.path.abspath(os.path.join(stock_data_path, f"{ticker}.NS.csv"))

        if not os.path.exists(stock_file):
            print(f"Stock data file not found: {stock_file}")
            continue

        # Read stock price data
        df_stock = pd.read_csv(stock_file)
        df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
        df_stock['Date'] = df_stock['Date'].dt.tz_localize(None)  # Ensure timezone-naive
        df_stock.set_index('Date', inplace=True)

        # Add technical indicators
        print("Adding technical indicators for", ticker)
        df_stock['SMA_20'] = ta.trend.sma_indicator(df_stock['Close'], window=20)
        df_stock['RSI_14'] = ta.momentum.rsi(df_stock['Close'], window=14)
        df_stock['Bollinger_High'] = ta.volatility.bollinger_hband(df_stock['Close'], window=20)
        df_stock['Bollinger_Low'] = ta.volatility.bollinger_lband(df_stock['Close'], window=20)
        df_stock.fillna(0, inplace=True)

        # Ensure only numeric columns are used for scaling
        print("Checking if technical indicators are present in", ticker)
        print(df_stock.head())
        numeric_columns = df_stock.select_dtypes(include=[np.number]).columns
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_stock[numeric_columns])
        scalers[ticker] = scaler

        # Prepare sequences for LSTM
        for i in range(sequence_length, len(scaled_data)):
            data.append((ticker, scaled_data[i-sequence_length:i], scaled_data[i, 0]))  # Predicting 'Close' price

        # Save the final DataFrame for verification
        df_stock.to_csv(os.path.join(stock_data_path, f"{ticker}_preprocessed.csv"))
        print(f"Saved preprocessed data for {ticker}")

    return data, scalers

if __name__ == "__main__":
    stock_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'stocks'))
    tickers = [
    "RELIANCE", "TCS", "INFY", "ICICIBANK", "ASIANPAINT", 
    "KOTAKBANK", "LT", "HINDUNILVR", "BAJFINANCE", "ITC",
    "BHARTIARTL", "SBIN", "MARUTI", "HDFCBANK",
    "ADANIGREEN", "WIPRO", "TECHM", "HCLTECH", "POWERGRID" ] 
    data, scalers = preprocess_stock_data(tickers, stock_data_path)
    print("Data preprocessing completed.")