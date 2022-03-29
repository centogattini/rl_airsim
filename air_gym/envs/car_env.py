#import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
#from airgym.envs.airsim_env import AirSimEnv
from simple_pid import PID
import matplotlib.pyplot as plt

class AirSimCarEnv(gym.Env):
    def __init__(self, model_name, test=False,contus=True):
        
        super().__init__()
        self.contus = contus
        self.test = test
        
        self.N_DOTS = 6
        self.SPEED = 6
        self.shape = (self.N_DOTS*2,)
        self.start_ts = 0
        self.observation_space = spaces.Box(-200, 200, shape=self.shape, dtype=np.float32)
        self.viewer = None

        self.model_name = model_name

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        self.car = airsim.CarClient(ip='127.0.0.1')
        self.road_arr = self._randomize_road()
        self.car.reset()
        if self.contus:
            self.action_space = spaces.Box(low=-1, high=1,shape=(1,),dtype='float32')
        else:
            self.action_space = spaces.Discrete(5)
        
        #self.image_request = airsim.ImageRequest(
        #    "0", airsim.ImageType.DepthPerspective, True, False
        #)

        self.car_controls = airsim.CarControls()
        self.car_state = None
        
        self.pid = PID(8,0.01,0.1,setpoint=self.SPEED)
        self.pid.output_limits = (0,1)
        
        self.reward = 0
        
        plt.ion()

        self.figure, self.ax = plt.subplots(figsize=(8,6))
        self.line1, = self.ax.plot([-20,20],[-20,20],'o-')
        self.ax.plot(0,0,'.',)
        plt.title("Dynamic Plot of road",fontsize=25)
        
        plt.xlabel("X",fontsize=18)
        plt.ylabel("Y",fontsize=18)
        
    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.02)

    def __del__(self):
        self.car.reset()
        
    def _do_action(self, action):
        throttle = self.pid(self.car.getCarState().speed)
        self.car_controls.throttle = throttle
        
        if self.contus:
            action = float(action)/2
            self.car_controls.steering = action
        else:
            if action == 0:
                self.car_controls.steering = 0
            elif action == 1:
                self.car_controls.steering = 0.5
            elif action == 2:
                self.car_controls.steering = -0.5
            elif action == 3:
                self.car_controls.steering = 0.2
            elif action == 4:
                self.car_controls.steering = -0.2
    
        self.car.setCarControls(self.car_controls)
        time.sleep(0.1)
    
    def _get_obs(self):

        self.car_state = self.car.getCarState()

        self.state['prev_pose'] = self.state['pose']
        self.state['pose'] = self.car_state.kinematics_estimated
        obs = np.array(self._get_pose()).flatten()
        #print(obs)
        #print(self.observation_space)
        return obs
    
    def _rotate(self,origin, points, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
    
        The angle should be given in radians.
        """
        ox, oy = [0,0]
        rotated_points = []
        for point in points:
            px, py = point
            qx = math.cos(angle) * (px-ox) - math.sin(angle) * (py-oy)
            qy = math.sin(angle) * (px-ox) + math.cos(angle) * (py-oy)
            rotated_points.append([qx,qy])
        return np.array(rotated_points)

    def _get_yaw(self,):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        kinematics = self.car.getCarState().kinematics_estimated
        orientation = kinematics.orientation
        x = orientation.x_val
        y = orientation.y_val
        z = orientation.z_val
        w = orientation.w_val

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.degrees(math.atan2(t3, t4))
    
        return yaw_z
       
    def _distance_to_road(self,car_pt):
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
        return dist
    
    def _get_car_position(self,):
        position = self.car.getCarState().kinematics_estimated.position
        x_pos = position.x_val
        y_pos = position.y_val
        car_pt = np.array([x_pos,y_pos])
        return car_pt
    
    def _compute_reward(self):
        
        BETA = -1/3
        
        car_pt = self._get_car_position()
        dist = self._distance_to_road(car_pt)
        reward = np.exp(dist*BETA) - 1/2
        #print('distance:',dist)
        #print('reward:',reward)
        done = 0
        if self.test == False:
            if dist >= 3:
                done = 1
        else:
            self._log_dist(dist)
            if dist >= 6:
                done = 1
        #print(dist, done)
        #if self.car_controls.brake == 0:
        #   if self.car_state.speed <= 0:
        #        done = 1
        if self.state["collision"]:
            done = 1
        
        self.reward+= reward
        
        return reward, done
    
    def _get_pose(self,):
        '''
        Input: car_state.kinematics_estimated
        Returns pose of veichle:
            
            next three dots of the route ///...
        ------- 
        
        '''
        pos = self._get_car_position()
        
        route = self._get_route(self.road_arr,pos,n=self.N_DOTS)
        transfered_route = np.array([x - pos for x in route])
        rotated_route = self._rotate(pos,transfered_route,-math.radians(self._get_yaw()))
        
        new_route = sorted(rotated_route, key=lambda x: x[0])

        x_dots = [r[0] for r in new_route]
        y_dots = [r[1] for r in new_route]
        self.line1.set_xdata(x_dots)
        self.line1.set_ydata(y_dots)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
        return new_route
    
    def _get_route(self,road_arr,car_coord,n=4):
        '''

        Parameters
        ----------
        road_arr : Road represented by array [mx2]
        car_coord : [x,y]-coordinates of our veichle
        n : TYPE, optional
            How many dots do we want
            DESCRIPTION. The default is 4.
        Returns
        -------
        np.array() with shape (n,2)
            Returns n closest to the veichle road spoints 

        '''
        n = self.N_DOTS
        car_coord = np.array(car_coord)
        road_arr = np.array(road_arr)
    
        sorted_arr = sorted(road_arr, key=lambda x: np.linalg.norm(x - car_coord))
        route = []
        for i in range(n):
            route.append(sorted_arr[i])
        
        return np.array(route)
    
    def _randomize_road(self,):
        from random import randrange
        from pandas import read_csv
        #n = randrange(4) + 1
        if self.test == False:
            m = randrange(6) + 1
            dots = read_csv(f'roads/gen_r/train/roads_data_{m}.csv')
            print('Test false')
        else:
            #n = randrange(6) + 1
            #dots = read_csv(f'roads/gen_r/test/roads_data_{n}.csv')
            dots = read_csv(f'roads/gen_r/long_test/roads_data_1.csv')
            print('Test true')
        road_arr = list(zip(dots['x'],dots['y']))
        return road_arr
    
    def set_test_mode(self,test_mode):
        self.test=test_mode
    
    def step(self, action):
        #print(action)
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        #print(obs, reward, bool(done), self.state)
        if self._check_road_end(obs):
            done = 1
        return obs, reward, bool(done), self.state
    
    def _check_road_end(self,obs):
        end = True
        for i in range(0,len(obs),2):
            if obs[i] >= 0:
                end = False
        return end
    
    def reset(self):
        self.road_arr = self._randomize_road()
        self._setup_car()
        self._do_action(0)
        print('reset!')
        print('Total reward:',self.reward)
        if self.test == False:
            self._log_rew()
        self.reward = 0
        return self._get_obs()
    
    def _log_route(self,):
        name = self.model_name
        f = open(f'learning_data/route_{name}.txt','a')
        f.write(f'{self._get_car_position()},')
        f.close()
    
    def _log_rew(self,):
        name = self.model_name
        f = open(f"learning_data/rewards_{name}.txt", "a")
        f.write(f'{self.reward},')
        f.close()
        
    def _log_dist(self,dist):
        name = self.model_name
        f = open(f"learning_data/dists_{name}.txt", "a")
        f.write(f'{dist},')
        f.close()
    def record(self, a):
        if a == True:
            self.car.startRecording()
        else:
            self.car.stopRecording()
           
    