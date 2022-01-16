#import gym
import numpy as np
import datetime
import time

# Всё необходимое для DDPG
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG

# Необходимое для PPO
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

# Необходимое для SAC
#from stable_baselines.sac.policies import MlpPolicy as MltPolicySAC
from stable_baselines3 import SAC

#import air_gym
from air_gym.envs.car_env import AirSimCarEnv


def train_DDPG(env, total_timesteps = 400000):
    """
    Тренируем DDPG

    env - наша среда, обёрнутая в OpenAI Gym

    total_timesteps - количество шагов обучения
    """
    #n_actions = env.action_space.shape[-1] # У нас есть пока только поворот, поэтому = 1
    n_actions = 1
    # the noise objects for DDPG
    param_noise = None
    #action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    
    # Выбираем модель и производим обучение
    model = DDPG(MlpPolicy, env, verbose=1,)
    #model = DDPG.load("DDPG_9_1_2022_19_11.zip") # если нам необходимо загрузить и дообучить
    start_time = time.time()
    model.learn(total_timesteps)
    learn_time = time.time()-start_time
    print(learn_time)
    dt_now = datetime.datetime.now()
    model.save(f'DDPG_{dt_now.day}_{dt_now.month}_{dt_now.year}_{dt_now.hour}_{dt_now.minute}')

    return model

def train_PPO(env, total_timesteps=25000, n_envs=4):
    """
    Тренируем PPO2 (Proximal Policy Optimization)

    n_envs - количество параллельных env

    total_timesteps - количество шагов обучения
    """
    # multiprocess environment
    env = make_vec_env(env, n_envs)

    model = PPO(MlpPolicy, env, verbose=1)
    # model = PPO2.load("path_to_model") # загружаем модель, если нужно дообучить

    model.learn(total_timesteps)

    dt_now = datetime.datetime.now()
    model.save(f'PPO2_{dt_now.day}_{dt_now.month}_{dt_now.year}_{dt_now.hour}_{dt_now.minute}')

    return model


def train_SAC(env, total_timesteps = 50000, log_interval=10):
    """
    Тренируем SAC (Soft Actor Critic)

    env - наша среда, обёрнутая в OpenAI Gym

    total_timesteps - количество шагов обучения

    log_interval - The number of timesteps before logging
    """

    # Выбираем модель и производим обучение
    model = SAC(MlpPolicy, env, verbose=1)
    # model = SAC.load("path_to_model") # если нам необходимо загрузить и дообучить

    model.learn(total_timesteps, log_interval)

    dt_now = datetime.datetime.now()
    model.save(f'SAC_{dt_now.day}_{dt_now.month}_{dt_now.year}_{dt_now.hour}_{dt_now.minute}')

    return model


def test_trained_agent(env, model):
    """
    Проверяем работу нашего агента.

    Возвращаем данные о наградах и наблюдениях
    """
    obs_data = []
    rewards_data = []

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        obs_data.append(obs)
        rewards_data.append(rewards)
        if dones:
            return rewards_data, obs_data
        # env.render() # это нам нужно?
        
if __name__ == '__main__':
    import pandas as pd
    dots = pd.read_csv('road_dots.csv')
    road_arr = list(zip(dots['0'],dots['1']))
    print('done')
    env = AirSimCarEnv(ip_address='127.0.0.1', road_arr=road_arr)
    model = train_DDPG(env,total_timesteps=4000)  
    #model = DDPG.load("DDPG_9_1_2022_19_11.zip")
    rewards_data, obs_data = test_trained_agent(env, model)
    print(rewards_data, obs_data)
    