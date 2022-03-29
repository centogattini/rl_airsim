import pandas as pd
models = ['DQN','DDPG','PPO','SAC','A2C']
for model in models:
    data = pd.read_csv(f'recs/{model}/airsim_rec.txt',sep='\t')
    x = data.POS_X
    y = data.POS_Y
    
    xy_data=list(zip(x,y))
    df = pd.DataFrame(xy_data)
    print(df)
    df.to_csv(f'road_dots_{model}.csv')
    print(df)