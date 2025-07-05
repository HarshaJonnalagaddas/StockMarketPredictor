import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close']]

# Function to prepare the dataset for training and testing
def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

# Function to split data into train and test sets
def split_data(X, y, train_ratio=0.8):
    train_size = int(len(X) * train_ratio)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# PyTorch dataset and dataloader class
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the model class
class StockModel(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, output_size, num_layers=1):
        super(StockModel, self).__init__()
        self.model_type = model_type
        if model_type == 'LSTM':
            self.model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'GRU':
            self.model = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'RNN':
            self.model = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'BiLSTM':
            self.model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size * (2 if model_type == 'BiLSTM' else 1), output_size)

    def forward(self, x):
        out, _ = self.model(x)
        if self.model_type == 'BiLSTM':
            out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.unsqueeze(-1))
            loss = criterion(outputs, y_batch.unsqueeze(-1))
            loss.backward()
            optimizer.step()

# Test the model
def test_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.unsqueeze(-1))
            predictions.append(outputs.numpy())
            actuals.append(y_batch.numpy())
    return np.concatenate(predictions), np.concatenate(actuals)

# Plot predictions
def plot_predictions(actual, predicted, model_type, ax, dates):
    ax.plot(dates, actual, label='Actual Prices', color='blue')
    ax.plot(dates, predicted, label=f'{model_type} Predictions', color='red')
    ax.set_title(f'{model_type} Stock Price Prediction')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

# Main program
def main():
    ticker = input("Enter the ticker symbol of the stock: ").upper()
    print(f"Fetching data for {ticker}...")

    # Fetch data
    data = fetch_stock_data(ticker, '2018-01-01', '2023-12-31')
    test_data = fetch_stock_data(ticker, '2024-01-01', '2025-02-23')

    # Prepare data
    X, y, scaler = prepare_data(data.values)
    X_train, X_test, y_train, y_test = split_data(X, y)

    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    models = ['LSTM', 'GRU', 'RNN', 'BiLSTM']
    hidden_size = 50
    input_size = 1
    output_size = 1
    num_layers = 2

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    dates = pd.date_range(start='2024-01-01', end='2025-02-23', periods=len(y_test))

    for i, model_type in enumerate(models):
        print(f"Training {model_type} model...")
        model = StockModel(model_type, input_size, hidden_size, output_size, num_layers=num_layers if model_type == 'BiLSTM' else 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, criterion, optimizer, num_epochs=10)

        predictions, actuals = test_model(model, test_loader)

        predictions = scaler.inverse_transform(predictions)
        actuals = scaler.inverse_transform(actuals.reshape(-1, 1))

        plot_predictions(actuals, predictions, model_type, axs[i], dates)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
