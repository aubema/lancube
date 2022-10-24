#!/usr/bin/env python3

# Remove multiple data from inventory
# Author : Julien-Pierre Houle

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import datetime


filename = 'lan3_invent_2022-02-08.csv'
df_inv = pd.read_csv(f'inventaires/{filename}')

radius = 10
max_diff_time = 90


transformer = Transformer.from_crs("EPSG:4326", "EPSG:2949", always_xy=True)
X, Y = transformer.transform(df_inv['lon_lights'], df_inv['lat_lights'])
# X, Y = transformer.transform(df_inv['lat_lights'], df_inv['lon_lights'])


coordinate = np.array(list(zip(X, Y)))
time = pd.to_datetime(df_inv['time']).values

total_idx = []

for idx, i in enumerate(coordinate):
    idx_to_delete = []

    idx = 0
    i = coordinate[idx]
    print(idx) 
       
    dx = i[0] - coordinate[idx+1:,0]
    dy = i[1] - coordinate[idx+1:,1]
    dist = np.sqrt(dx**2 + dy**2)
    
    idx_close = np.where(dist < radius)
    tech0 = df_inv['tech'].iloc[idx]
        
    for j in idx_close[0]:
            
        diff_time = time[idx] - time[j]
        diff_time = abs(diff_time.astype('timedelta64[s]').astype(np.int32))
        
        if tech0 == df_inv['tech'].iloc[j] and diff_time > max_diff_time:
            idx_to_delete.append(j)
            
    if len(idx_to_delete) > 0:
        all_idx = np.concatenate([ [idx], idx_to_delete ])
        
        df_inv.loc[idx, 'lat_lights'] = np.mean( df_inv['lat_lights'].iloc[all_idx] )
        df_inv.loc[idx, 'lon_lights']  = np.mean( df_inv['lon_lights'].iloc[all_idx] )
        
        df_inv.loc[idx, 'H'] = np.mean( df_inv['H'].iloc[all_idx] )
        df_inv.loc[idx, 'd'] = np.mean( df_inv['d'].iloc[all_idx] )
        
        df_inv.loc[idx, 'flux'] = np.mean( df_inv['flux'].iloc[all_idx] )
        df_inv.loc[idx, 'lux'] = np.mean( df_inv['lux'].iloc[all_idx] )
    
    total_idx.append(idx_to_delete)

total_idx = [i for sublist in total_idx for i in sublist]
idx_to_delete = np.unique(np.array(total_idx))

mask = np.ones(len(df_inv), dtype=bool)
mask[idx_to_delete] = False


# Update dataframe
df_inv = df_inv[mask]
df_inv.to_csv(f'inventaires/{filename}_update.csv', index=False)
