# AI Crypto Market Predictor (Multi-Coin, 15-Min Interval, Classification + Alerts + Web App Ready)

import pandas as pd
import numpy as np
import requests
import argparse
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import json
import os
import streamlit as st

# === Step 1: Download Historical Price Data ===
def fetch_data(coin="bitcoin", vs_currency="usd", interval="minutely", points=100):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency={vs_currency}&days=1&interval=minute"
    response = requests.get(url)
    data = response.json()
    prices = data['prices'][-points:]  # Only keep latest N points
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# === Step 2: Prepare Data for Classification ===
def prepare_classification_data(df, window_size=15):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['price']])
    X, y = [], []
    for i in range(window_size, len(scaled_data) - 1):
        X.append(scaled_data[i - window_size:i, 0])
        # Binary classification: 1 if price goes up next, else 0
        y.append(int(scaled_data[i + 1, 0] > scaled_data[i, 0]))
    return np.array(X), np.array(y), scaler

# === Step 3: Build and Train Classification LSTM Model ===
def build_classification_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Step 4: Send Webhook/Alert Notification ===
def send_alert(message, webhook_url=None, save_to_file=True):
    if webhook_url:
        try:
            headers = {'Content-Type': 'application/json'}
            payload = json.dumps({"content": message})
            requests.post(webhook_url, headers=headers, data=payload)
        except Exception as e:
            print(f"[ERROR] Failed to send webhook: {e}")
    if save_to_file:
        log_path = os.path.expanduser("~/crypto_signals.log")
        with open(log_path, "a") as f:
            f.write(message + "\n")

# === Step 5: Main Pipeline ===
def run(coin='bitcoin', webhook_url=None, confidence_threshold=0):
    df = fetch_data(coin)
    X, y, scaler = prepare_classification_data(df)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = build_classification_model((X.shape[1], 1))
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_class)

    latest_X = X[-1].reshape(1, X.shape[1], 1)
    latest_pred = model.predict(latest_X)[0][0]
    direction = "UP" if latest_pred > 0.5 else "DOWN"
    confidence = round(latest_pred * 100 if direction == "UP" else (1 - latest_pred) * 100, 2)

    message = f"[{datetime.utcnow()} UTC] {coin.upper()} 15-min Prediction: {direction} | Confidence: {confidence}% | Model Accuracy: {round(acc*100, 2)}%"

    if confidence >= confidence_threshold:
        print(message)
        send_alert(message, webhook_url)
    else:
        print(f"Prediction skipped due to low confidence ({confidence}%)")

    return message, direction, confidence, acc

# === Streamlit Web App ===
def dashboard():
    st.title("🧠 AI Crypto Market Predictor")
    coin = st.selectbox("Select a Coin", ["bitcoin", "ethereum", "solana", "ripple", "cardano", "dogecoin", "binancecoin", "avalanche-2", "chainlink"])
    confidence_filter = st.slider("Confidence Threshold %", 0, 100, 0)
    if st.button("Run Prediction"):
        message, direction, confidence, acc = run(coin=coin, confidence_threshold=confidence_filter)
        st.success(message)
        st.metric("Direction", direction)
        st.metric("Confidence", f"{confidence}%")
        st.metric("Model Accuracy", f"{round(acc*100, 2)}%")

# === CLI Interface ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crypto AI Market Movement Predictor')
    parser.add_argument('--coin', type=str, default='bitcoin', help='Cryptocurrency name (e.g., bitcoin, ethereum, solana)')
    parser.add_argument('--webhook', type=str, help='Webhook URL for sending alerts (optional)')
    parser.add_argument('--threshold', type=float, default=0, help='Minimum confidence % required to trigger alert')
    parser.add_argument('--dashboard', action='store_true', help='Launch Streamlit dashboard')
    args = parser.parse_args()

    if args.dashboard:
        dashboard()
    else:
        run(coin=args.coin, webhook_url=args.webhook, confidence_threshold=args.threshold)

# === To schedule this every 15 minutes on macOS or Linux ===
# 1. Run `crontab -e`
# 2. Add the following line:
# */15 * * * * /usr/bin/python3 /full/path/to/ai_crypto_predictor.py --coin bitcoin --webhook https://your.webhook.url --threshold 70 >> /path/to/log.txt 2>&1

# === To launch the dashboard ===
# streamlit run /full/path/to/ai_crypto_predictor.py
