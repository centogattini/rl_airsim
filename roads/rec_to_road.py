import pandas as pd
for model in ['dqn']:
    data = pd.read_csv(f'airsim_recs/airsim_rec_final.txt',sep='\t')
    x = data.POS_X
    y = data.POS_Y
    #x_ = []
    #y_ = []
    #for j in range(1,len(x),20):
    #    x_.append(x[j])
    #    y_.append(y[j])
    xy_data=list(zip(x,y))
    df = pd.DataFrame(xy_data)
    df.to_csv(f'road_dots_final.csv')
    print(df)