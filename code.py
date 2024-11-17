import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Step 1: Fetch Historical Stock Data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)
    return data

# Step 2: Preprocess Data
def preprocess_data(data):
    data['Day'] = np.arange(len(data))
    X = data[['Day']]  # Feature: Day index
    y = data['Close']  # Target: Closing price
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Make Predictions
def predict(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Step 5: Visualize Results
def visualize_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.reset_index(drop=True), label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.show()

# Main Program
if __name__ == "__main__":
    # Configuration
    TICKER = "AAPL"  # Example: Apple Inc.
    START_DATE = "2020-01-01"
    END_DATE = "2023-01-01"

    # Fetch and preprocess data
    stock_data = fetch_data(TICKER, START_DATE, END_DATE)
    X_train, X_test, y_train, y_test = preprocess_data(stock_data)

    # Train model
    model = train_model(X_train, y_train)

    # Predict and evaluate
    predictions = predict(model, X_test)
    print("Mean Squared Error:", mean_squared_error(y_test, predictions))

    # Visualize results
    visualize_results(y_test, predictions)
