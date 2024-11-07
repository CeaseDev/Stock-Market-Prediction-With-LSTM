import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

# Load the preprocessed data from CSV files

def load_train_data(tickers, data_path, holdout_days=10):
    X_train = []
    y_train = []
    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()  # Separate scaler for the target column
    sequence_length = 60

    for ticker in tickers:
        file_path = os.path.join(data_path, f"{ticker}_preprocessed.csv")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        df.dropna(inplace=True)

        # Select only numeric columns, excluding 'Date' and other non-numeric columns
        df = df.select_dtypes(include=[np.number])
        df[df.columns] = scaler.fit_transform(df[df.columns])
        df['Close'] = target_scaler.fit_transform(df[['Close']])
        data = df.values

        # Reserve the last `holdout_days` for testing and use the rest for training
        train_data = data[:-holdout_days]

        # Prepare training data
        for i in range(sequence_length, len(train_data)):
            X_train.append(train_data[i-sequence_length:i])
            y_train.append(train_data[i, 3])  # Assuming 'Close' is at index 3

    X_train = np.array(X_train, dtype='float32')
    y_train = np.array(y_train, dtype='float32')

    return X_train, y_train, scaler, target_scaler

if __name__ == "__main__":
    # Specify the path to the CSV files
    data_path = '/mnt/e/Stock_prediction_major/data/stocks'
    tickers = [
        "RELIANCE", "TCS", "INFY", "ICICIBANK", "ASIANPAINT", 
        "KOTAKBANK", "LT", "HINDUNILVR", "BAJFINANCE", "ITC",
        "BHARTIARTL", "SBIN", "MARUTI", "HDFCBANK",
        "ADANIGREEN", "WIPRO", "TECHM", "HCLTECH", "POWERGRID"
    ]

    # Load training data
    X_train, y_train, scaler, target_scaler = load_train_data(tickers, data_path)

    # Reshape data for LSTM (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))

    # Split data into training and validation sets
    split_index = int(0.8 * len(X_train))
    X_train, X_val = X_train[:split_index], X_train[split_index:]
    y_train, y_val = y_train[:split_index], y_train[split_index:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Save the model
    model.save('/mnt/e/Stock_prediction_major/models/lstm_stock_model.h5')
    print("Model training completed and saved.")

    # Optionally, save training history
    with open('/mnt/e/Stock_prediction_major/models/training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)
