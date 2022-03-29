import numpy as np
import datetime
import time

# Всё необходимое для DDPG
#from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from air_gym.envs.car_env import AirSimCarEnv
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

def create_model_and_env(model_name):
    if model_name == 'SAC':
        env = AirSimCarEnv(model_name=m_n,test=True)
        model = SAC.load('learning_data/SAC/SAC_epoch_40.zip', env, verbose=1)
        return model, env
    
    elif model_name == 'A2C':
        env = AirSimCarEnv(model_name=m_n,test=True)
        model = A2C.load('learning_data/A2C/A2C_epoch_78.zip', env)
        return model, env
    
    elif model_name == 'DDPG':
        env = AirSimCarEnv(model_name=m_n,test=True)
        model = DDPG.load('learning_data/DDPG/DDPG_epoch_68.zip',env)
        return model, env
    
    elif model_name == 'PPO':
        env = AirSimCarEnv(model_name=m_n,test=True)
        model = PPO.load('learning_data/PPO/PPO_epoch_73.zip', env)
        return model, env
    
    elif model_name == 'DQN':
        env = AirSimCarEnv(model_name=m_n,contus=False,test=True)
        model = DQN.load("learning_data/DQN/DQN_epoch_76.zip", env, verbose=1)
        return model, env
    else: 
        return None
        
def evaluate_model(env,model,model_name,epoch):
    storage = []
    
    env.set_test_mode(True)
    
    obs = env.reset()
    obs_data = []
    rewards_data = []
    
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        obs_data.append(obs)
        storage.append(rewards)
        rewards_data.append(rewards)
        if dones:
            min_rew = min(rewards_data)
            min_dist = np.log(min_rew + 1/2)*(-3)
            sum_rew = sum(rewards_data)
                
            f = open(f"learning_data/{model_name}/validation_data_{epoch}.txt", "a")
            f.write(f'\'model\': {model_name}\n \n \'rewards_data\': {rewards_data},\n \'min_dist\':{min_dist} \n \'sum_of_rews\':{sum_rew}\n,\'len\':{len(rewards_data)}\n')
            f.close()
            break
    mean_reward = sum(storage)/5
    
    f = open(f'learning_data/{model_name}/validation_data_mean_{epoch}.txt','a')
    f.write(f'{mean_reward},')
    f.close()
    env.set_test_mode(False)
    return mean_reward

def test_trained_agent(env, model,model_name):
    """
    Проверяем работу нашего агента.

    Возвращаем данные о наградах и наблюдениях
    """
    obs_data = []
    rewards_data = []
    
    obs = env.reset()
    start_time = time.time()
    
    env.record(True)
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        obs_data.append(obs)
        rewards_data.append(rewards)
        if dones:
            env.record(False)
            min_rew = min(rewards_data[:-5])
            max_dist = np.log(min_rew + 1/2)*(-3)
            
            mean_rew = np.mean(rewards_data[:-5])
            mean_dist = np.log(mean_rew + 1/2)*(-3)
            
            
            #sum_rew = sum(rewards_data)
            
            learn_time = time.time()-start_time
            print(f'{model_name} time: {learn_time}, max distance: {max_dist}, mean distance: {mean_dist}')
            return rewards_data, obs_data
            

if __name__ == '__main__':
    model_names = ['DQN','DDPG','PPO','SAC','A2C']
    for m_n in model_names:
        model,env = create_model_and_env(m_n)
        env.reset()
        test_trained_agent(env,model,m_n)