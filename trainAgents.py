    #import gym
import numpy as np
import datetime
import time

# Всё необходимое для DDPG
#from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise

from stable_baselines3.common.env_checker import check_env
# Необходимое для PPO
# from stable_baselines.common.policies import MlpPolicy
#from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import TD3
from stable_baselines3 import DQN
from stable_baselines3 import SAC
# Необходимое для SAC
#from stable_baselines.sac.policies import MlpPolicy as MltPolicySAC


#import air_gym
from air_gym.envs.car_env import AirSimCarEnv
from air_gym.envs.car_env_discrete import AirSimCarEnvDiscrete


def train_DDPG(env, total_timesteps = 100000):
    
    action_noise = NormalActionNoise(mean=0, sigma=float(0.001))
    model = DDPG('MlpPolicy',env,verbose=1,learning_starts=100,
                 action_noise=action_noise,batch_size=64,
                 tau=0.001,learning_rate=0.0001)
    
    start_time = time.time()
    model.learn(total_timesteps)
    learn_time = time.time()-start_time
    print(learn_time)
    dt_now = datetime.datetime.now()
    model.save(f'DDPG_{dt_now.day}_{dt_now.month}_{dt_now.year}_{dt_now.hour}_{dt_now.minute}')

    return model

def train_PPO(env, total_timesteps=25000):

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps)
    dt_now = datetime.datetime.now()
    model.save(f'PPO_{dt_now.day}_{dt_now.month}_{dt_now.year}_{dt_now.hour}_{dt_now.minute}')

    return model


def train_SAC(env,model,total_timesteps = 50000):
    
    #model = SAC('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    dt_now = datetime.datetime.now()
    model.save(f'models/SAC_{dt_now.day}_{dt_now.month}_{dt_now.year}_{dt_now.hour}_{dt_now.minute}')

    return model

def train_A2C(env,total_timesteps = 50000):
    
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps)
    dt_now = datetime.datetime.now()
    
    model.save(f'A2C_{dt_now.day}_{dt_now.month}_{dt_now.year}_{dt_now.hour}_{dt_now.minute}')
    
    return model

def train_TD3(env,total_timesteps=100000):
    action_noise = NormalActionNoise(mean=0, sigma=float(0.001))
    model = TD3('MlpPolicy',env,verbose=1,learning_starts=100,
                 action_noise=action_noise,batch_size=64,
                 tau=0.001,learning_rate=0.0001)
    model.learn(total_timesteps)
    dt_now = datetime.datetime.now()
    model.save(f'TD3_{dt_now.day}_{dt_now.month}_{dt_now.year}_{dt_now.hour}_{dt_now.minute}')
    
    return model
    
def train_DQN(env,total_timesteps=100000):
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps)
    
    dt_now = datetime.datetime.now()
    model.save(f'DQN_{dt_now.day}_{dt_now.month}_{dt_now.year}_{dt_now.hour}_{dt_now.minute}')
    
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
        
    
def create_model(model_name,env):
    if model_name == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1)
        return model
    elif model_name == 'A2C':
        model = A2C("MlpPolicy", env, verbose=1)
        return model
    elif model_name == 'DDPG':
        action_noise = NormalActionNoise(mean=0, sigma=float(0.001))
        model = DDPG('MlpPolicy',env,verbose=1,learning_starts=100,
                     action_noise=action_noise,batch_size=64,
                     tau=0.001,learning_rate=0.0001)
        return model
    elif model_name == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1)
        return model
    elif model_name == 'DQN':
        model = DQN("MlpPolicy", env, verbose=1)
        return model
    else: 
        return None
    

def evaluate_model(env,model,model_name):
    obs_data = []
    rewards_data = []
    storage = []
    env.set_test_mode(set_test_mode=True)
    obs = env.reset()
    for i in range(5):
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            obs_data.append(obs)
            rewards_data.append(rewards)
            if dones:
                min_rew = min(rewards_data)
                min_dist = np.log(min_rew + 1/2)*(-3)
                    
                f = open(f"{}_validation_data_final.txt", "a")
                f.write(f'rewards_data: {rewards_data},\n time:{end_time} \n min dist:{min_dist} \n sum of rews:{sum(rewards_data)}')
                f.close()
    

if __name__ == '__main__':
    
    model_names = ['SAC','DDPG','A2C','PPO']
    n_epochs = 10
    total_timesteps = 10000
    for m in model_names:
        
        model = create_model(m)
        for i in n_epochs:
            env = AirSimCarEnv(model_name=m)
            model.learn(total_timesteps=total_timesteps,reset_num_timesteps=False)
            env = AirSimCarEnv(model_name=m,test=True)
            
        
    
    
    env = AirSimCarEnv(model_name='SAC')
    
    start_time = time.time()
    model = train_SAC(env,total_timesteps=100000)
    learn_time = time.time()-start_time
    
    rewards_data, obs_data = test_trained_agent(env, model)
    f = open("SAC_rewards_data.txt", "a")
    f.write(f'rewards_data: {rewards_data} \n time:{learn_time}')
    f.close()
    
    del env
    '''
    env = AirSimCarEnv(model_name='DQN',contus=False)
    start_time = time.time()
    model = train_DQN(env,total_timesteps=100000)
    learn_time = time.time()-start_time
    
    rewards_data, obs_data = test_trained_agent(env, model)
    f = open("DQN_rewards_data.txt", "a")
    f.write(f'rewards_data: {rewards_data} \n time:{learn_time}')
    f.close()
   
    env = AirSimCarEnvDiscrete(model_name='TD3')
    start_time = time.time()
    model = train_DQN(env,total_timesteps=100000)
    learn_time = time.time()-start_time
    
    rewards_data, obs_data = test_trained_agent(env, model)
    f = open("DQN_rewards_data.txt", "a")
    f.write(f'rewards_data: {rewards_data} \n time:{learn_time}')
    f.close()
    '''
    
    '''
    
    env = AirSimCarEnv(model_name='DDPG')
    start_time = time.time()
    model = train_DDPG(env,total_timesteps=100000)
    learn_time = time.time()-start_time
    
    rewards_data, obs_data = test_trained_agent(env, model)
    f = open("DDPG_rewards_data.txt", "a")
    f.write(f'rewards_data: {rewards_data} \n time:{learn_time}')
    f.close()
    
    env = AirSimCarEnv(model_name='PPO')
    
    start_time = time.time()
    model = train_PPO(env,total_timesteps=100000)
    learn_time = time.time()-start_time
    
    rewards_data, obs_data = test_trained_agent(env, model)
    f = open("PPO_rewards_data.txt", "a")
    f.write(f'rewards_data: {rewards_data} \n time:{learn_time}')
    f.close()
    
    del env
    
    env = AirSimCarEnv(model_name='SAC')
    
    start_time = time.time()
    model = train_SAC(env,total_timesteps=100000)
    learn_time = time.time()-start_time
    
    rewards_data, obs_data = test_trained_agent(env, model)
    f = open("SAC_rewards_data.txt", "a")
    f.write(f'rewards_data: {rewards_data} \n time:{learn_time}')
    f.close()
    
    del env
    
    env = AirSimCarEnv(model_name='A2C')
    
    start_time = time.time()
    model = train_A2C(env,total_timesteps=100000)
    learn_time = time.time()-start_time
    
    rewards_data, obs_data = test_trained_agent(env, model)
    f = open("A2C_rewards_data.txt", "a")
    f.write(f'rewards_data: {rewards_data} \n time:{learn_time}')
    f.close()
    '''