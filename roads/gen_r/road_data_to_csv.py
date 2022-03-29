import json
import pandas as pd
for i in range(1,2):
    j_f = open(f'train/roads_data_{i}.txt')
    vrs = json.load(j_f)
    j_f.close()
    x = vrs['roads_x']
    y = vrs['roads_y']
    d = {'x':x,'y':y}
    df = pd.DataFrame(d)
    df.to_csv(f'train/roads_data_{i}.csv')
