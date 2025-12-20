# metrics_emitter.py
from prometheus_client import start_http_server, Gauge
import time
import random

RPS = Gauge('app_requests_per_second', 'Requests per second')
LAT = Gauge('app_latency_ms', 'Latency in ms')
CPU = Gauge('app_cpu_usage', 'CPU usage percent')

def simulate_metrics():
    # Simple workload pattern: baseline + sinusoid + noise + occasional spike
    t = 0.0
    while True:
        baseline = 50
        rps = baseline + 20 * (0.5 + 0.5 * random.random()) * (1 + 0.5 * random.choice([0,1]))
        if random.random() < 0.02:  # spike
            rps *= random.uniform(2.5, 5.0)
        latency = max(10, 200 * (rps / 100.0)) + random.gauss(0,5)
        cpu = min(95, 20 + rps * 0.2 + random.gauss(0,2))

        RPS.set(rps)
        LAT.set(latency)
        CPU.set(cpu)

        t += 1
        time.sleep(1)

if __name__ == '__main__':
    start_http_server(8000)
    print("Metrics endpoint started on http://0.0.0.0:8000/metrics")
    simulate_metrics()

