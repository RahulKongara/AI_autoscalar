# env_gym.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ScalingEnv(gym.Env):
    """
    State: [current_rps, current_pods, avg_latency]
    Action: discrete: 0=down, 1=no-op, 2=up
    Reward: negative cost + penalty for high latency
    """

    def __init__(self, max_pods=20, target_latency=200):
        super().__init__()
        self.max_pods = max_pods
        self.target_latency = target_latency

        # Observation space
        obs_low  = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        obs_high = np.array([1000.0, float(max_pods), 10000.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action space: 0 = down, 1 = no-op, 2 = up
        self.action_space = spaces.Discrete(3)

        # Internal state
        self.pods = None
        self.rps = None
        self.latency = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pods = 2
        self.rps = 50.0
        self.latency = 100.0

        obs = np.array([self.rps, self.pods, self.latency], dtype=np.float32)
        info = {}

        return obs, info

    def step(self, action):
        # Action mapping
        if action == 0 and self.pods > 1:
            self.pods -= 1
        elif action == 2 and self.pods < self.max_pods:
            self.pods += 1

        # Simulate workload variation
        self.rps = max(0, self.rps + np.random.randn()*5 +
                       np.random.choice([0, 50], p=[0.98, 0.02]))

        # Latency model
        base = 50 + 0.5 * self.rps
        self.latency = base / max(1, self.pods) * (1 + np.random.randn()*0.05)

        # Cost model
        cost = self.pods * 0.1

        # Reward formula
        penalty = max(0, (self.latency - self.target_latency) / self.target_latency) * 10
        reward = -cost - penalty

        # Episode termination
        terminated = False
        truncated = False  # or use a max_steps counter if needed

        obs = np.array([self.rps, self.pods, self.latency], dtype=np.float32)

        info = {"cost": cost, "latency": self.latency}

        return obs, reward, terminated, truncated, info
