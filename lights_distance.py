# Calculate the lights distances between each other
# Julien-Pierre Houle
# Mai 2022


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def distance(lat_obs, lon_obs, arr_lats, arr_lon):
    return np.sqrt((lat_obs - arr_lats)**2+((lon_obs-arr_lon)*np.cos(arr_lon))**2)


def find_close_lights(df, df_invent, nb=10, h=2):    
    
    lat, lon = [], []
    update_H = []
    for i in df.index.values:
        
        lat_obs = df_invent['lat_lights'].iloc[i]
        lon_obs = df_invent['lon_lights'].iloc[i]
        
        list_dist = []
        for j in range(len(df_invent)):
            lat_lights = df_invent['lat_lights'].iloc[j]
            lon_lights = df_invent['lon_lights'].iloc[j]

            list_dist.append(distance(lat_obs, lon_obs, lat_lights, lon_lights))
        
        arr_dist = np.array(list_dist)
        idx_dist = np.argwhere([arr_dist < 0.005])
        idx_dist = idx_dist[idx_dist != 0]
        
        lat.append(df_invent['lat_lights'].iloc[idx_dist].values)
        lon.append(df_invent['lon_lights'].iloc[idx_dist].values)
        
        # Mean of the 10 closest lights
        df = df_invent.iloc[idx_dist]
        
        # on considere lights seulement > 4m
        update_H.append( df['H'][(df['H'] > 4) & (df['flux'] < 30000)].mean() )

    return update_H, lat, lon
    


def filter_multip_detections(df_invent):

    list_duplicat = []

    for i in df_invent.index.values:
        
        if i not in list_duplicat:
            # print('liste duplicat:', list_duplicat)
            # print('i:', i)
            lat_obs = df_invent['lat_lights'].iloc[i]
            lon_obs = df_invent['lon_lights'].iloc[i]
            H_obs = df_invent['H'].iloc[i]
            tech_obs = df_invent['tech'].iloc[i]
            
            # Calcul distance from obs light fixtures
            list_dist = []
            for j in range(len(df_invent)):
                # if j not in list_duplicat:
                    lat_lights = df_invent['lat_lights'].iloc[j]
                    lon_lights = df_invent['lon_lights'].iloc[j]
                    list_dist.append(distance(lat_obs, lon_obs, lat_lights, lon_lights))
            
            arr_dist = np.array(list_dist)
            idx_dist = np.argwhere(arr_dist < 0.0001).flatten()
            df = df_invent.iloc[idx_dist]
            
            # Look for similar H and tech
            df = df[(df['H'] > H_obs*0.5) & (df['H'] < H_obs*1.5)]
            df = df[df['tech'] == tech_obs]
            df = df[~np.in1d(df.index.values, list_duplicat)] # remove if index in duplicat
            # print('df index values:', df.index.values)

            
            # Average the lights caract
            df_invent['H'].iloc[i] = df['H'].mean()
            df_invent['flux'].iloc[i] = df['flux'].mean()
            df_invent['lat_lights'].iloc[i] = df['lat_lights'].mean()
            df_invent['lon_lights'].iloc[i] = df['lon_lights'].mean()
            df_invent['R/G'].iloc[i] = df['R/G'].mean()
            df_invent['B/G'].iloc[i] = df['B/G'].mean()
                
            for k in df.index.tolist():
                if k != i:
                    list_duplicat.append(k)
                    
    # Remove duplicat
    list_duplicat = np.unique(list_duplicat)
    df_invent = df_invent.drop(list_duplicat).reset_index(drop=True)

    return df_invent
