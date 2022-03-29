import pandas as pd

data = pd.read_csv('airsim_rec_big_road.txt',sep='\t')
qx = data.POS_X
qy = data.POS_Y
x = data.POS_X
y = data.POS_Y
x_ = []
y_ = []
for i in range(1,len(x),30):
    x_.append(x[i])
    y_.append(y[i])
xy_data=list(zip(x_,y_))
df = pd.DataFrame(xy_data)
df.to_csv('road_dots_big_road.csv')
