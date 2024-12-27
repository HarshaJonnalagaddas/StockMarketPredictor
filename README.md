# Stock Market Prediction using RNN - Jonnalagadda Sri Harsha

## Introduction
This repository contains two implementations for stock market prediction using Recurrent Neural Networks (RNN). The files include:

1. **`stock.ipynb`**: A sample notebook designed for understanding stock market data, performing Exploratory Data Analysis (EDA), and applying basic machine learning techniques.
2. **`sl_model.py`**: A generalized solution for stock prediction using a dynamic Bidirectional LSTM model, enabling improved training and evaluation.

## Background

### Recurrent Neural Networks (RNN)
RNNs are specialized neural networks for sequential data. They allow information to persist by looping through the network, making them suitable for time-series prediction tasks.

### Long Short-Term Memory (LSTM)
LSTMs are an advanced variant of RNNs that address the problem of long-term dependency. By utilizing gates (input, forget, and output), LSTMs retain relevant past information effectively.

### Bidirectional LSTM
A Bidirectional LSTM processes input sequences in both forward and backward directions, capturing dependencies from past and future states. This dual approach improves the model's prediction accuracy.

---

## File Descriptions

### 1. `stock.ipynb`
This notebook:
- Explores stock market data.
- Performs data cleaning and visualization.
- Implements basic data preprocessing techniques for sequential modeling.
- Provides insights into feature engineering and initial modeling.

This file is intended for those new to stock prediction and data analysis, serving as a hands-on introduction.

### 2. `sl_model.py`
This Python script:
- Implements a dynamic Bidirectional LSTM model to predict stock prices.
- Provides a streamlined and generalized solution for training RNN models on different stocks.
- Includes the following features:
  - Data preprocessing for individual company stock data.
  - Training and testing set creation with time-step-based sequences.
  - Bidirectional LSTM model design with Dropout layers to reduce overfitting.
  - EarlyStopping for efficient training.
  - Visualization of prediction results.

---

## Initialization and Requirements

### Prerequisites
- Install Python 3.8 or above.
- Install required libraries using:

```bash
pip install -r requirements.txt
```

### Dataset
The dataset used for these implementations is `all_stocks_5yr.csv`. It contains historical stock prices for various companies over five years.

### Usage
1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_name>
```

2. Run `stock.ipynb` for initial data exploration and analysis.

3. Execute `sl_model.py` for end-to-end stock prediction. The script prompts you to select a company for prediction.

---

## Future Plans
1. **Web Application**: Integrate the model into a web application for real-time stock predictions.
2. **Technical Indicators**: Enhance model input with popular technical indicators like RSI, MACD, and Bollinger Bands for improved accuracy.
3. **Portfolio Analysis**: Extend the model to analyze and predict portfolios of stocks.

---

## Contribution
Feel free to contribute by raising issues or submitting pull requests. Suggestions for improvement and collaboration are welcome!

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
