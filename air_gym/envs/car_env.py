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
    def __init__(self, ip_address, road_arr):
        
        super().__init__()
        self.N_DOTS = 4
        self.SPEED = 7
        self.shape = (self.N_DOTS,2)
        self.start_ts = 0
        self.observation_space = spaces.Box(-200, 200, shape=self.shape, dtype=np.float32)
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
        
        self.road_arr = road_arr
        
        self.pid = PID(8,0.01,0.1,setpoint=self.SPEED)
        self.pid.output_limits = (0,1)
        
        plt.ion()
        x = np.linspace(0, 10, 100)
        y = np.cos(x)
        self.figure, self.ax = plt.subplots(figsize=(8,6))
        self.line1, = self.ax.plot([-20,20],[-20,20],'.-')
        plt.title("Dynamic Plot of road",fontsize=25)
        
        plt.xlabel("X",fontsize=18)
        plt.ylabel("Y",fontsize=18)
        
    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()
        
    def _do_action(self, action):
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
        self.car.setCarControls(self.car_controls)
        
        #debug information
        {
            print('Steering:',action)
            
        }
        time.sleep(1)
        
    def _get_obs(self):

        self.car_state = self.car.getCarState()

        self.state['prev_pose'] = self.state['pose']
        self.state['pose'] = self.car_state.kinematics_estimated
        
        return self._get_pose()
    
    def _rotate(self,origin, points, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
    
        The angle should be given in radians.
        """
        ox, oy = [0,0]
        rotated_points = []
        for point in points:
            px, py = point
            qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
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
        
        BETA = 0.5
        
        car_pt = self._get_car_position()
        dist = self._distance_to_road(car_pt)
 
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
        pos = self._get_car_position()
        
        route = self._get_route(self.road_arr,pos)
        transfered_route = np.array([x - pos for x in route])
        rotated_route = self._rotate(pos,transfered_route,math.radians(self._get_yaw()))
        
        new_route = rotated_route

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
    
    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        print(done, bool(done))
        return obs, reward, bool(done), self.state

    def reset(self):
        self._setup_car()
        self._do_action([[0]])
        print('reset!')
        return self._get_obs()
