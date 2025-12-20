# train_forecast.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

def create_dataset(series, seq_len=30):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

# load CSV of historical rps (or generate quickly)
# here we simulate data:
def generate_series(n=5000):
    import math, random
    s=[]
    for t in range(n):
        base = 50 + 10*math.sin(2*math.pi*t/144)  # daily seasonality
        noise = random.gauss(0,4)
        spike = random.choice([0,0,0, 0, 30 if random.random()<0.01 else 0])
        s.append(max(0, base + noise + spike))
    return np.array(s)

series = generate_series(4000)
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1,1)).flatten()
joblib.dump(scaler, 'scaler_rps.pkl')

SEQ_LEN = 30
X, y = create_dataset(series_scaled, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(64, input_shape=(SEQ_LEN,1), activation='tanh'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.1)

model.save('lstm_rps.h5')
print("Saved lstm_rps.h5 and scaler_rps.pkl")

