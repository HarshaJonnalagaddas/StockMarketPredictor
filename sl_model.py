# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

data = pd.read_csv('all_stocks_5yr.csv', delimiter=',', on_bad_lines='skip')
data['date'] = pd.to_datetime(data['date'])  # Convert date column to datetime

def preprocess_company_data(data, company_name):
    # Filter data for the selected company
    company_data = data[data['Name'] == company_name].sort_values(by='date')
    
    # Scale the 'close' prices
    company_data['close_scaled'] = (company_data['close'] - company_data['close'].min()) / (
        company_data['close'].max() - company_data['close'].min())
    train_size = int(len(company_data) * 0.8)
    train_data = company_data['close_scaled'].iloc[:train_size].values
    test_data = company_data['close_scaled'].iloc[train_size:].values
    
    # Function to create input sequences for RNN
    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:i + time_step])
            y.append(data[i + time_step])
        return np.array(X), np.array(y)
    
    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    # Reshape inputs for RNN model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return company_data, X_train, y_train, X_test, y_test
def build_and_train_bi_rnn(X_train, y_train, X_test, y_test):
    from tensorflow.keras.callbacks import EarlyStopping
    
    model = tf.keras.Sequential([
        # Bidirectional RNN with 64 units
        tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64, return_sequences=True, input_shape=(X_train.shape[1], 1))),
        tf.keras.layers.Dropout(0.2),  # Dropout to prevent overfitting
        
        # Second Bidirectional RNN layer
        tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64, return_sequences=False)),
        tf.keras.layers.Dropout(0.2),  # Dropout again
        tf.keras.layers.Dense(1)
    ])
     
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Add EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping])
    
    return model


def predict_and_visualize(model, company_data, X_train, X_test, y_train, y_test, time_step=60):
    # Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    min_close = company_data['close'].min()
    max_close = company_data['close'].max()
    train_predict = train_predict * (max_close - min_close) + min_close
    test_predict = test_predict * (max_close - min_close) + min_close
    y_train_actual = y_train * (max_close - min_close) + min_close
    y_test_actual = y_test * (max_close - min_close) + min_close

    plt.figure(figsize=(10, 6))
    plt.plot(company_data['date'], company_data['close'], label='Actual Prices', color='blue')
    plt.plot(company_data['date'].iloc[time_step:time_step + len(train_predict)], train_predict, label='Train Predictions', color='green')
    plt.plot(company_data['date'].iloc[-len(test_predict):], test_predict, label='Test Predictions', color='red')
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"Stock Price Predictions for {company_data['Name'].iloc[0]}")
    plt.legend()
    plt.show()

def display_companies_grouped(data):
    unique_companies = sorted(data['Name'].unique())  # Get unique, sorted company names
    grouped_companies = {}

    # Group compinies by their first letter
    for company in unique_companies:
        key = company[0].upper()
        if key not in grouped_companies:
            grouped_companies[key] = []
        grouped_companies[key].append(company)

    # Display grouped companies
    print("Available Companies (Grouped by First Letter):")
    for letter, companies in grouped_companies.items():
        print(f"\n{letter}:")  
        print(", ".join(companies))

display_companies_grouped(data)
# User input to select a company
selected_company = input("Enter the company name (from the list above): ").strip()
if selected_company not in data:
    print("Invalid company name. Please try again.")
company_data, X_train, y_train, X_test, y_test = preprocess_company_data(data, selected_company)
model = build_and_train_bi_rnn(X_train, y_train, X_test, y_test)
predict_and_visualize(model, company_data, X_train, X_test, y_train, y_test)


