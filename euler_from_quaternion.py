import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
def euler_from_quaternion(x, y, z, w):
  """
  Convert a quaternion into euler angles (roll, pitch, yaw)
  roll is rotation around x in radians (counterclockwise)
  pitch is rotation around y in radians (counterclockwise)
  yaw is rotation around z in radians (counterclockwise)
  """
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll_x = math.degrees(math.atan2(t0, t1))

  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch_y = math.degrees(math.asin(t2))

  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z = math.degrees(math.atan2(t3, t4))

  return roll_x, pitch_y, yaw_z # in radians

def rotate(origin, points, angle):
    """
        Rotate a point counterclockwise by a given angle around a given origin.
    
        The angle should be given in radians.
    """
    ox, oy = origin
    rotated_points = []
    x_ = []
    y_ = []
    for point in points:
        px, py = point
        qx = math.cos(angle) * (px-ox) - math.sin(angle) * (py-oy)
        qy = math.sin(angle) * (px-ox) + math.cos(angle) * (py-oy)
        rotated_points.append((qx,qy))
        x_.append(qx)
        y_.append(qy)
    return x_,y_

if __name__ == '__main__':
    df = pd.read_csv('airsim_rec.txt',sep='\t')
    x_pos = df.POS_X
    y_pos = df.POS_Y
    q_w = df.Q_W
    q_x = df.Q_X
    q_y = df.Q_Y
    q_z = df.Q_Z
    x = []
    for i in range(1,len(x_pos),1):
        x.append((x_pos[i],y_pos[i],q_w[i],q_x[i],q_y[i],q_z[i]))
    arr = []
    for i in range(len(q_w)):
        x_,y_,z_ = euler_from_quaternion(q_x[i], q_y[i], q_z[i], q_w[i])
        arr.append((math.radians((z_)),(x_pos[i],y_pos[i])))
        
    #plt.plot(arr)
    #plt.show()
    
    plt.ion()

    figure, ax = plt.subplots(figsize=(8,6))
    line1, = ax.plot([-10,100],[-100,10],'.-')
    ax.plot(0,0,'.',)
    plt.title("Dynamic Plot of road",fontsize=25)
        
    plt.xlabel("X",fontsize=18)
    plt.ylabel("Y",fontsize=18)
    
    dots = pd.read_csv('road_dots.csv')
    road_arr = list(zip(dots['0'],dots['1']))
    for angle, pos in arr:
        r = rotate(pos,road_arr,-angle)
        x_dots,y_dots = r
        line1.set_xdata(x_dots)
        line1.set_ydata(y_dots)
        #ax.plot(pos[0],,'o')
        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.01)