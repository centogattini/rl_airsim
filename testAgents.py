# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:54:06 2022

@author: user
"""
from air_gym.envs.car_env import AirSimCarEnv
#from air_gym.envs.car_env_discrete import AirSimCarEnvDiscrete
#import datetime
import time
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
from stable_baselines3 import DQN
import numpy as np
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
        

if __name__ == '__main__':

    env = AirSimCarEnv(model_name='SAC',test=True)

    model = SAC.load('models/sac/SAC_27_2_2022_20_0.zip', env)
    
    start_time = time.time()
    rewards_data, obs_data = test_trained_agent(env, model)
    end_time = time.time()-start_time
    
    min_rew = min(rewards_data)
    min_dist = np.log(min_rew + 1/2)*(-3)
    
    f = open("SAC_validation_data_final.txt", "a")
    f.write(f'rewards_data: {rewards_data},\n time:{end_time} \n min dist:{min_dist} \n sum of rews:{sum(rewards_data)}')
    f.close()
    
    '''
    env = AirSimCarEnv(model_name='DDPG',test=True)

    model = DDPG.load('DDPG_23_2_2022_22_20.zip', env)
    
    start_time = time.time()
    rewards_data, obs_data = test_trained_agent(env, model)
    end_time = time.time()-start_time
    
    min_rew = min(rewards_data)
    min_dist = np.log(min_rew + 1/2)*(-3)
    
    f = open("DDPG_validation_data.txt", "a")
    f.write(f'rewards_data: {rewards_data},\n time:{end_time} \n min dist:{min_dist} \n sum of rews:{sum(rewards_data)}')
    f.close()
    
    
    env = AirSimCarEnv(model_name='PPO',test=True)

    model = PPO.load('PPO_21_2_2022_15_0.zip', env)
    
    start_time = time.time()
    rewards_data, obs_data = test_trained_agent(env, model)
    end_time = time.time()-start_time
    
    min_rew = min(rewards_data)
    min_dist = np.log(min_rew + 1/2)*(-3)
    
    f = open("PPO_validation_data.txt", "a")
    f.write(f'rewards_data: {rewards_data},\n time:{end_time} \n min dist:{min_dist} \n sum of rews:{sum(rewards_data)}')
    f.close()
    
    del env
    
    env = AirSimCarEnv(model_name='SAC',test=True)
    
    model = SAC.load('SAC_21_2_2022_22_36.zip',env)
    
    start_time = time.time()
    rewards_data, obs_data = test_trained_agent(env, model)
    end_time = time.time()-start_time
    
    min_rew = min(rewards_data)
    min_dist = np.log(min_rew + 1/2)*(-3)
    
    f = open("SAC_validation_data.txt", "a")
    f.write(f'rewards_data: {rewards_data},\n time:{end_time}\n min dist:{min_dist} \n sum of rews:{sum(rewards_data)}')
    f.close()
    
    del env
    
    env = AirSimCarEnv(model_name='A2C',test=True)
    
    model = A2C.load('A2C_22_2_2022_4_53.zip',env)
    
    start_time = time.time()
    rewards_data, obs_data = test_trained_agent(env, model)
    end_time = time.time()-start_time
    
    min_rew = min(rewards_data)
    min_dist = np.log(min_rew + 1/2)*(-3)
    
    f = open("A2C_validation_data.txt", "a")
    f.write(f'rewards_data: {rewards_data},\n time:{end_time} \n min dist:{min_dist} \n sum of rews:{sum(rewards_data)}')
    f.close()
    '''