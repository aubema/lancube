# Calculate the lights distances between each other
# Julien-Pierre Houle
# Mai 2022


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def distance(lat_obs, lon_obs, arr_lats, arr_lon):
    return np.sqrt((lat_obs - arr_lats)**2+((lon_obs-arr_lon)*np.cos(arr_lats*(np.pi/180)))**2)


def find_close_lights(df, df_invent, nb=10, h=2):
    """ Find lights close to problematic sources to average height """ 
    
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
        idx_dist = np.argwhere([arr_dist < 0.05])
        idx_dist = idx_dist[idx_dist != 0]
        
        lat.append(df_invent['lat_lights'].iloc[idx_dist].values)
        lon.append(df_invent['lon_lights'].iloc[idx_dist].values)
        
        df = df_invent.iloc[idx_dist]
        
        # on considere lights seulement > 4m
        update_H.append( df['H'][(df['H'] > 4) & (df['flux'] < 30000)].mean() )
            
        df_coord = pd.DataFrame({'lat':lat, 'lon':lon})
        
    return update_H, df_coord




def bearing_calculation(lat1, lon1, lat2, lon2):
    
    dlon = lon2- lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    brng = np.arctan2(y, x)
    brng = np.rad2deg(brng)
    brng = (brng + 360) % 360
    brng = 360 - brng  # count degrees counter-clockwise (remove to make clockwise)
    
    return brng


    


def filter_multip_detections(df_invent, df_initial, prec_localisation, condition_side=True):

    lat, lon = [], []
    list_duplicat = []
    theta_min = (prec_localisation * 180) / (np.pi * 6373000)  # Limite par le GPS
    
    
    # Calculer bearing pour tous les peaks
    arr_brng = np.zeros(len(df_invent))
    for j, i in enumerate(df_invent.index.values):
        
        # Retrouver la valeur correspondante dans la df original du Lan3
        df_corresp = df_initial.loc[(df_initial['lat'] == (df_invent.iloc[i]['lat_peaks'])) &\
                                    (df_initial['lon'] == (df_invent.iloc[i]['lon_peaks']))]

        # Trouver les datas initiales prec 5 secondes
        time_point = df_corresp['Time'].iloc[0]
        df_time_rng = df_initial[(df_initial['Time'] < time_point) & 
                                 (df_initial['Time'] > time_point-pd.Timedelta(5,'seconds'))]
        df_time_rng.sort_values(by='Time') # sort pour etre chronologique
        
        lat1, lon1 = df_time_rng['lat'].iloc[0], df_time_rng['lon'].iloc[0] #position 5sec avant
        lat2, lon2 = df_corresp['lat'].iloc[0], df_corresp['lon'].iloc[0]
        
        arr_brng[j] = bearing_calculation(lat1, lon1, lat2, lon2)
        
    df_invent['bearing'] = arr_brng
    
    
    

    for i in df_invent.index.values:
        
        if i not in list_duplicat:
            lat_obs = df_invent['lat_lights'].iloc[i]
            lon_obs = df_invent['lon_lights'].iloc[i]
            H_obs = df_invent['H'].iloc[i]
            tech_obs = df_invent['tech'].iloc[i]
            side_obs = df_invent['side'].iloc[i]
            flux_obs = df_invent['flux'].iloc[i]
            df_original = df_invent.loc[i].to_frame().T
            bearing_obs = df_invent['bearing'].iloc[i]
            
            
            # Calcul distance from obs light fixtures
            list_dist = []
            for j in range(len(df_invent)):
                lat_lights = df_invent['lat_lights'].iloc[j]
                lon_lights = df_invent['lon_lights'].iloc[j]
                list_dist.append(distance(lat_obs, lon_obs, lat_lights, lon_lights))
            
            arr_dist = np.array(list_dist)
            idx_dist = np.argwhere(arr_dist < theta_min).flatten()
            df = df_invent.iloc[idx_dist]  
            
            
            # Look for similar H and tech
            if condition_side:
                df = df[(df['H'] > H_obs*0.5) & (df['H'] < H_obs*3)]
            else:
                df = df[(df['H'] > H_obs*0.75) & (df['H'] < H_obs*1.25)]


            df = df[df['tech'] == tech_obs]
            df = df[~np.in1d(df.index.values, list_duplicat)] # remove if index in duplicat
            
                 
            # Check time difference
            if condition_side:
                idx_filter = []
                for j in range(len(df)):
                    timediff = abs(df['time'].iloc[j] - df_invent['time'].iloc[i])
                    if timediff.seconds >= 60:
                        idx_filter.append(j)
                df_time = df.iloc[idx_filter]  # keep lights with time > 60s
                
            else:
                df_time = df

            df = pd.concat([df_original, df_time]) # concat init obs. and other with time > 60s
            
            
            # Regarder orientation des vecteurs deplacement
            if condition_side == True:
                if len(df) > 1:
                                
                    direction_opo = np.zeros(len(df), dtype=bool)
                    diff_bearing = np.abs(df['bearing'].values - bearing_obs)
                    
                    idx_opo = np.argwhere(diff_bearing > 90)  
                    direction_opo[idx_opo] = True
                            

                    # Check side of detection (different than the original)
                    # df = df.reset_index(drop=True)
                    idx_to_delete = []
                    for idx, j in enumerate(df.index.values):
                        
                        if direction_opo[idx] == True:                        
                            if df.iloc[idx]['side'] == side_obs:
                                idx_to_delete.append(j)                              
                        else:
                            if df.iloc[idx]['side'] != side_obs:
                                idx_to_delete.append(j)
                    
                    df = df.drop(idx_to_delete)
                

     
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
                
    df_invent = df_invent.drop(list_duplicat).reset_index(drop=True)

    return df_invent





def filter_small(df_small, df_invent, prec_localisation):
    """ Remove small lights close to big lights with same tech """
    
    lat, lon = [], []
    theta_min = (prec_localisation * 180) / (np.pi * 6373000)     # Limite par le GPS
    flux_perc = 0.5
    idx_small_drop = []

    for i in df_small.index.values:
        lat_obs = df_small['lat_lights'].iloc[i]
        lon_obs = df_small['lon_lights'].iloc[i]
        
        list_dist = []
        for j in range(len(df_invent)):
            lat_lights = df_invent['lat_lights'].iloc[j]
            lon_lights = df_invent['lon_lights'].iloc[j]

            list_dist.append(distance(lat_obs, lon_obs, lat_lights, lon_lights))
        
        arr_dist = np.array(list_dist)
        idx_dist = np.argwhere([arr_dist < theta_min])
        idx_dist = idx_dist[idx_dist != 0]
        
               
        df = df_invent.iloc[idx_dist] # lights close to small lights
        df = df[ df['tech'] == df_small['tech'].iloc[i] ] # same tech
        df = df[df['H'] > 4] # hauteur minimal de 4m
        df = df[ df_small['flux'].iloc[i] < df['flux']*flux_perc ]
        
        if len(df) > 0: # Si il y a un grand lampadaire qui rempli les conditions
            idx_small_drop.append(i)
            lat.append(df_small['lat_lights'].iloc[i].tolist())
            lon.append(df_small['lon_lights'].iloc[i].tolist())
               
    df_small = df_small.drop(idx_small_drop)
    df_coord = pd.DataFrame({'lat':lat, 'lon':lon})
    
    return df_small, df_coord
