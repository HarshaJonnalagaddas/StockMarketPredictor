# Stock Market Prediction using RNNs and FinBERT - Jonnalagadda Sri Harsha

## 🔍 Introduction
This repository provides a comprehensive exploration of deep learning models for stock market prediction. It includes:

- Exploratory data analysis and traditional forecasting methods.
- Implementation and **comparative analysis** of RNN-based models: **RNN**, **LSTM**, **GRU**, and **Bidirectional LSTM (BiLSTM)**.
- A hybrid **BiLSTM + FinBERT** architecture integrating sentiment analysis from real-time financial news headlines.

---

## 📁 Project Structure

| File/Notebook       | Description |
|---------------------|-------------|
| `stock.ipynb`       | Notebook for stock market EDA, visualizations, and basic modeling. |
| `sl_model.py`       | Dynamic BiLSTM-based stock price prediction model. |
| `compare.py`        | **Comparative analysis** of RNN, LSTM, GRU, and BiLSTM models using PyTorch. |
| `fibertLSTM.py`     | **Hybrid model** combining BiLSTM with FinBERT-based news sentiment for enhanced prediction. |

---

## 🧠 Background

### ⏱ Recurrent Neural Networks (RNNs)
RNNs process sequential data, making them ideal for time-series tasks such as stock prediction. However, they struggle with long-term dependencies.

### 💡 LSTM and GRU
- **LSTM (Long Short-Term Memory)** introduces memory gates to capture long-range dependencies effectively.
- **GRU (Gated Recurrent Unit)** is a simplified version of LSTM with similar performance but fewer parameters.

### 🔁 Bidirectional LSTM
Processes sequences in both forward and backward directions, capturing a richer context.
![image](https://github.com/user-attachments/assets/51488648-ade0-4f0e-80b4-b33297455268)
### 📰 FinBERT for Sentiment
[FinBERT](https://huggingface.co/ProsusAI/finbert) is a pre-trained NLP model tuned on financial texts. It helps capture **sentiment polarity** from recent news headlines, integrated into stock price predictions.

---

## 📊 Comparative Model Insights

Run `compare.py` to visualize and compare the performance of:

- ✅ RNN
- ✅ LSTM
- ✅ GRU
- ✅ BiLSTM

Each model is trained on the same dataset and their predictions are plotted side by side for evaluation.

---

## 🧠 FinBERT + BiLSTM Hybrid (fibertLSTM.py)

This advanced script:
- Fetches the latest stock prices (2019–2025).
- Gathers recent news headlines using `StockNews` RSS.
- Applies FinBERT sentiment scoring to each headline.
- Integrates sentiment with BiLSTM predictions using a linear combination.
- Visualizes:
  - BiLSTM-only predictions
  - BiLSTM + FinBERT-enhanced predictions

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt


## Contribution
Feel free to contribute by raising issues or submitting pull requests. Suggestions for improvement and collaboration are welcome!

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
