import os
import yfinance as yf
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from transformers import pipeline
from stocknews import StockNews

os.environ["TRANSFORMERS_NO_TF"] = "1"
def fetch_stock_data(ticker):
    data = yf.download(ticker, start="2019-01-01", end="2025-02-01")
    data = data[["Close"]]
    return data
def preprocess_data(data, scaler):
    scaled_data = scaler.fit_transform(data)
    x, y = [], []
    for i in range(60, len(scaled_data)):
        x.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(x), np.array(y)
class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    return model

def predict(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x_batch, _ in data_loader: 
            print(f"x_batch before processing: {type(x_batch)}, shape: {x_batch.shape if isinstance(x_batch, torch.Tensor) else 'Not a Tensor'}")
            
            if len(x_batch.shape) == 2:  
                x_batch = x_batch.unsqueeze(-1)  

            print(f"x_batch after processing: {x_batch.shape}")
            outputs = model(x_batch)
            predictions.append(outputs.numpy())
    return np.concatenate(predictions).reshape(-1)

def fetch_recent_news(ticker):
    sn = StockNews(ticker, save_news=False)
    news_data = sn.read_rss()
    headlines = news_data['title'].tolist()
    timestamps = news_data['published'].tolist()
    return headlines, pd.to_datetime(timestamps)
def analyze_sentiment(news):
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt", device=0 if torch.cuda.is_available() else -1)
    sentiments = sentiment_pipeline(news)
    sentiment_scores = []
    for sentiment in sentiments:
        if sentiment["label"] == "positive":
            sentiment_scores.append(1)
        elif sentiment["label"] == "negative":
            sentiment_scores.append(-1)
        else:
            sentiment_scores.append(0)
    return np.array(sentiment_scores)

def aggregate_sentiment(news, timestamps):
    sentiment_scores = analyze_sentiment(news)
    sentiment_df = pd.DataFrame({"timestamp": timestamps, "headline": news, "sentiment": sentiment_scores})
    print("\nNews Headlines and Sentiment Scores:")
    print(sentiment_df[["timestamp", "headline", "sentiment"]])  # Print news and scores
    daily_sentiment = sentiment_df.groupby("timestamp")["sentiment"].mean()
    return daily_sentiment
def combine_predictions(bilstm_predictions, sentiment_scores, alpha=0.1):
    sentiment_effect = alpha * sentiment_scores[:len(bilstm_predictions)]
    combined_predictions = bilstm_predictions + sentiment_effect
    return combined_predictions

def plot_results(actual_prices, bilstm_predictions, combined_predictions):
    plt.figure(figsize=(20, 12))  # Further increased figure size

    # BiLSTM-only predictions
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(actual_prices, label="Actual Prices", color="blue")
    ax1.plot(bilstm_predictions, label="BiLSTM Predictions", color="red")
    ax1.set_title("BiLSTM Stock Price Prediction", fontsize=18)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Price", fontsize=12)

    # BiLSTM + Sentiment predictions
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(actual_prices, label="Actual Prices", color="blue")
    ax2.plot(combined_predictions, label="BiLSTM + Sentiment Predictions", color="green")
    ax2.set_title("BiLSTM + Sentiment Stock Price Prediction", fontsize=18)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Price", fontsize=12)

    plt.tight_layout()
    plt.show()

def main():
    ticker = input("Enter stock ticker: ")
    stock_data = fetch_stock_data(ticker)

    # BiLSTM setup
    scaler = MinMaxScaler()
    x_data, y_data = preprocess_data(stock_data.values, scaler)
    x_train, y_train = x_data[:-300], y_data[:-300]
    x_test, y_test = x_data[-300:], y_data[-300:]

    # Convert data to tensors with correct shapes
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32)  # Convert y_test to tensor

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(x_test, torch.zeros_like(x_test))  # Dummy targets for DataLoader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BiLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = train_model(model, train_loader, criterion, optimizer, epochs=20)

    bilstm_predictions = predict(model, test_loader)
    bilstm_predictions = scaler.inverse_transform(bilstm_predictions.reshape(-1, 1))
    actual_prices = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))
    news, timestamps = fetch_recent_news(ticker)
    daily_sentiment = aggregate_sentiment(news, timestamps)
    combined_predictions = combine_predictions(bilstm_predictions, daily_sentiment.values.flatten())
    plot_results(actual_prices, bilstm_predictions, combined_predictions)


if __name__ == "__main__":
    main()

