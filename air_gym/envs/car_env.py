#import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
#from airgym.envs.airsim_env import AirSimEnv
from simple_pid import PID

class AirSimCarEnv(gym.Env):
    def __init__(self, ip_address, road_arr):
        
        super().__init__()

        self.shape = (3,2)
        self.start_ts = 0
        self.observation_space = spaces.Box(-200, 200, shape=self.shape, dtype=np.uint8)
        self.viewer = None

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        self.car = airsim.CarClient(ip=ip_address)
        self.car.reset()
        self.action_space = spaces.Box(low=-0.5, high=0.5 ,shape=(1,1),dtype=np.float32)
        
        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, True, False
        )

        self.car_controls = airsim.CarControls()
        self.car_state = None
        
        self.N_DOTS = 3
        self.SPEED = 9
        self.road_arr = road_arr
        
        self.pid = PID(8,0.01,0.1,setpoint=9)
        self.pid.output_limits = (0,1)
        
    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()
        
    def _do_action(self, action):
        print('Steering:',action)
        try:
            action = action[0]
            try:
                action = action[0]
            except:
                pass
        except:
            pass
        action = float(action)
        self.car_controls.steering = action

        throttle = self.pid(self.car.getCarState().speed)
        self.car_controls.throttle = throttle
        print('throttle:',throttle)
        self.car.setCarControls(self.car_controls)
        time.sleep(1)
        
    def _get_obs(self):

        self.car_state = self.car.getCarState()

        self.state['prev_pose'] = self.state['pose']
        self.state['pose'] = self.car_state.kinematics_estimated
        
        return self._get_pose()

    def _compute_reward(self):
        #MAX_SPEED = 300
        #MIN_SPEED = 10
        #THRESH_DIST = 5
        BETA = 0.5
        
        #pts = pts_alt
        position = self.car.getCarState().kinematics_estimated.position
        x_pos = position.x_val
        y_pos = position.y_val
        car_pt = np.array([x_pos,y_pos])
        
        
        dist = 10000000
        
        pts = self._get_route(self.road_arr,car_pt,2)
        
        # compute a distance between car position and a line between road dots
        pts = np.array(pts)
        car_pt = np.array(car_pt)
        dist = min(
                dist,
                np.linalg.norm(
                    np.cross((car_pt - pts[0]), (pts[0] - pts[1]))
                )
                / np.linalg.norm(pts[0] - pts[1])
        )
        
        reward = math.exp(-BETA * dist) - 0.5
        print('distance:',dist)
        print('reward:',reward)
        done = 0
        if reward <= -0.27:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 0:
                done = 1
        if self.state["collision"]:
            done = 1
        
        return reward, done
    
    def _get_pose(self,):
        '''
        Input: car_state.kinematics_estimated
        Returns pose of veichle:
            
            next three dots of the route ///...
        ------- 
        
        '''
        position = self.car.getCarState().kinematics_estimated.position
        x_pos = position.x_val
        y_pos = position.y_val
        pos = np.array([x_pos,y_pos])
        route = self._get_route(self.road_arr,pos)
        new_route = np.array([x - pos for x in route])
        return new_route
    
    def _get_route(self,road_arr,car_coord,n=3):
        
        car_coord = np.array(car_coord)
        road_arr = np.array(road_arr)
        sorted_arr = sorted(road_arr, key=lambda x: np.linalg.norm(x - car_coord))
        route = []
        for i in range(n):
            route.append(sorted_arr[i])
        return np.array(route)
    
    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action([[0]])
        print('reset!')
        return self._get_obs()
