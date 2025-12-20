# forecast.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import requests
import time

SCALER = 'scaler_rps.pkl'
MODEL = 'lstm_rps.h5'
SEQ_LEN = 30

def fetch_recent_rps(prometheus_url='http://localhost:8000/metrics'):
    # For prototype using metrics emitter, scrape metrics endpoint simply:
    # In production, query Prometheus API for the last seq_len points.
    # Here: call the metric endpoint and parse -- but simpler: use random or file
    import re
    r = requests.get(prometheus_url)
    m = re.search(r'app_requests_per_second (\d+(\.\d+)?)', r.text)
    if m:
        return float(m.group(1))
    return 50.0

def predict_next(n_steps=5):
    scaler = joblib.load(SCALER)
    model = load_model(MODEL)

    window = []
    for _ in range(SEQ_LEN):
        window.append(fetch_recent_rps())
        time.sleep(0.1)
    arr = scaler.transform(np.array(window).reshape(-1,1)).flatten()
    seq = arr.copy()
    preds = []
    for _ in range(n_steps):
        x = seq[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        p = model.predict(x)[0,0]
        preds.append(float(scaler.inverse_transform([[p]])[0,0]))
        seq = np.append(seq, p)
    return preds

if __name__ == "__main__":
    print("Next 5-step predictions:", predict_next(5))

