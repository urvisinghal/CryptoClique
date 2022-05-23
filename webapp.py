import streamlit as st
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta as rd

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.keras.models import Sequential

start = (date.today() - rd(years=1)).strftime("%Y-%m-%d")
end =  date.today().strftime("%Y-%m-%d")

st.image("https://cdn.discordapp.com/attachments/977301415645032532/977661652348600350/unknown-modified_1.png", width=200)
st.title("CryptoClique")

crypto = ("BTC-USD", "ETH-USD", "BNB-USD", "DOGE-USD", "USDT-USD", "USDC-USD", "XRP-USD", "HEX-USD", "BUSD-USD", "ADA-USD", "SOL-USD", "DOT-USD", "WBTC-USD", "AVAX-USD", "WTRX-USD", "TRX-USD", "STETH-USD", "DAI-USD", "SHIB-USD", "MATIC-USD", "LTC-USD", "CRO-USD", "LEO-USD", "YOUC-USD", "NEAR-USD")
crypto_selected = st.selectbox("Select the currency for prediction", crypto)

@st.cache
def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

data = load_data(crypto_selected).sort_values(by='Date',ascending=False)

st.subheader("Dataset")
st.write(data.head())

#Visualisation
st.subheader("Visualisation")
def plot_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
    fig.layout.update(title_text="Opening and Closing vs Time", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_data()


#normalisation
scaler = MinMaxScaler()
close_price = data.Close.values.reshape(-1, 1)
scaled_close = scaler.fit_transform(close_price)
scaled_close = scaled_close.reshape(-1, 1)


#preprocessing
SEQ_LEN = 20

def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split = 0.75)


#model
DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1

model = keras.Sequential()

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True),
                        input_shape=(WINDOW_SIZE, X_train.shape[-1])))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))

model.add(Dense(units=1))

model.add(Activation('linear'))


#training
model.compile(
    loss='mean_squared_error', 
    optimizer='adam',
)

BATCH_SIZE = 64

history = model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    validation_split=0.1
)


#prediction
y_hat = model.predict(X_test)

y_test_inverse = (scaler.inverse_transform(y_test)).flatten()
y_hat_inverse = (scaler.inverse_transform(y_hat)).flatten()

st.subheader("CryptoCurrency Price Predictor")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data["Date"], y=y_test_inverse, name="Actual Price"))
fig2.add_trace(go.Scatter(x=data["Date"], y=y_hat_inverse, name="Predicted Price"))
fig2.layout.update(title_text="Original vs Prediction", xaxis_rangeslider_visible=True)
st.plotly_chart(fig2)