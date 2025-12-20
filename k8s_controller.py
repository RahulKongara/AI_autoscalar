# k8s_controller.py
import time
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from kubernetes import client, config
from stable_baselines3 import PPO

# Config
NAMESPACE = 'default'
DEPLOYMENT_NAME = 'demo-app'
MIN_REPLICAS = 1
MAX_REPLICAS = 20
COOLDOWN = 30  # seconds
USE_RL = True  # if True use RL policy; else use forecast thresholds

# Load models
if not USE_RL:
    scaler = joblib.load('scaler_rps.pkl')
    lstm = load_model('lstm_rps.h5')
else:
    rl_policy = PPO.load('ppo_scaler')

# Kubernetes config (works inside cluster automatically; locally use kubeconfig)
try:
    config.load_incluster_config()
except:
    config.load_kube_config()
apps_v1 = client.AppsV1Api()

last_action_time = 0
def get_current_rps():
    # Query Prometheus or scrape emitter; here a simple placeholder
    import requests, re
    try:
        r = requests.get('http://localhost:8000/metrics', timeout=1)
        m = re.search(r'app_requests_per_second (\d+(\.\d+)?)', r.text)
        if m: return float(m.group(1))
    except Exception:
        pass
    return 50.0

def scale_deployment(replicas):
    replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, int(replicas)))
    body = {'spec': {'replicas': replicas}}
    apps_v1.patch_namespaced_deployment_scale(DEPLOYMENT_NAME, NAMESPACE, body)
    print(f"[Controller] Scaled {DEPLOYMENT_NAME} -> {replicas} replicas")

def get_deployment_replicas():
    dep = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
    return dep.spec.replicas

def run_loop():
    global last_action_time
    while True:
        rps = get_current_rps()
        # Simple forecast-based rule
        if not USE_RL:
            # (For demo, use current rps to pick replicas = ceil(rps/50))
            desired = max(MIN_REPLICAS, min(MAX_REPLICAS, int(np.ceil(rps / 50.0))))
        else:
            # RL policy: state = [rps, current_replicas, last_latency]
            cur_reps = get_deployment_replicas() or 1
            # For latency placeholder use rps/cur_reps
            latency = (rps / max(1, cur_reps)) * 2
            state = np.array([rps, cur_reps, latency], dtype=np.float32)
            action, _ = rl_policy.predict(state, deterministic=True)
            # action: 0->down,1->noop,2->up
            if action == 0:
                desired = max(MIN_REPLICAS, cur_reps - 1)
            elif action == 2:
                desired = min(MAX_REPLICAS, cur_reps + 1)
            else:
                desired = cur_reps

        # cooldown
        if time.time() - last_action_time > COOLDOWN:
            cur = get_deployment_replicas()
            if desired != cur:
                scale_deployment(desired)
                last_action_time = time.time()
            else:
                print(f"[Controller] No scaling needed. replicas={cur}")
        else:
            print("[Controller] Cooling down...")

        time.sleep(10)

if __name__ == '__main__':
    run_loop()

