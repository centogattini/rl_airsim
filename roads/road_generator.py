import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

def next_dot(a,b,s):
    a = np.array(a)
    b = np.array(b)
    dist = random.random()*8+4
    max_angle = (1.42 - 0.17*dist)/4
    angle = random.uniform(-max_angle,max_angle)    
    
    ab_v = (b-a)/np.linalg.norm(b-a)# vector ab
    d = b + (np.cos(angle)*dist)*ab_v
    p_ab_v = np.array([-ab_v[1],ab_v[0]])/np.linalg.norm(np.array([-ab_v[1],ab_v[0]]))
    c = dist*np.sin(angle)*p_ab_v*random.choice([-1,1]) + d
    s+=dist
    return c,s

if __name__ == '__main__':

    a,b = ((0,0),(1,0))
    roads_dots=[]
    roads_x=[]
    roads_y=[]
    j=1 
    while True:
        x=[]
        y=[]
        dots = [a,b]           
        s=0
        for i in range(100):
            d1 = dots[i]
            d2 = dots[i+1]
            d3,s_s = next_dot(d1,d2,s)
            s=s_s
            dots.append(d3)
            x.append(d3[0])
            y.append(d3[1])
        print(s)
        if np.abs(s-800)<=3:
            plt.plot(x,y,'.-')
            plt.show()
            inp = input(f'good road number {j}? type y/n/end:')        
            if inp =='y':
                roads_dots.append(dots)
                roads_x.append(x)
                roads_y.append(y)
                s_i = '{'+f'"roads_x":{x},\n "roads_y":{y}'+'}'
                d = pd.DataFrame({'x':x,'y':y})
                d.to_csv(f'gen_r/long_test/roads_data_{j}.csv')
                j+=1
            elif inp == 'end':
                break
        
            
    

