# train_rl.py
import gymnasium as gym
from stable_baselines3 import PPO
from env_gym import ScalingEnv

def train():
    env = ScalingEnv()
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=200_000)
    model.save('ppo_scaler')
    print("Saved ppo_scaler.zip")

if __name__ == '__main__':
    train()

