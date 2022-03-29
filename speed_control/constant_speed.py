
import airsim

#!/usr/bin/env python

import time
import matplotlib.pyplot as plt
from simple_pid import PID
if __name__ == '__main__':
        client = airsim.CarClient()
        client.reset()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        car_controls = airsim.CarControls()

        pid = PID(8, 0.01, 0.1, setpoint=9)
        pid.output_limits = (0, 80)

        start_time = time.time()
        last_time = start_time

        # Keep track of values for plotting
        setpoint, y, x = [], [], []

        while time.time() - start_time < 10:
            current_time = time.time()
            speed = client.getCarState().speed
            power = pid(speed)
            car_controls.throttle = power
            x += [current_time - start_time]
            y += [speed]
            setpoint += [pid.setpoint]
            print(power)
            client.setCarControls(car_controls)
            last_time = current_time
            time.sleep(1)
        plt.plot(x, y, label='measured')
        plt.plot(x, setpoint, label='target')
        plt.xlabel('time')
        plt.ylabel('temperature')
        plt.legend()
        plt.show()
