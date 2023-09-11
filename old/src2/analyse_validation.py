# Analyse the accuracy of the Lan3 predictions
# Julien-Pierre Houle
# June 2022

import numpy as np
import pandas as pd


# Load Ground Inventory
PATH = '/home/jhoule42/Documents/Lancube'
df_inv_stcamille = pd.read_csv(f'{PATH}/Validation/inv_stcamille.csv')
df_inv_stoke = pd.read_csv(f'{PATH}/Validation/inv_stoke.csv')

# Load Lan3 Inventory
df_lan3_stcamille = pd.read_csv(f'{PATH}/Validation/lancube_stcamille.csv')
df_lan3_stoke = pd.read_csv(f'{PATH}/Validation/lancube_stoke.csv')



def distance(lat_obs, lon_obs, arr_lats, arr_lon):
    return np.sqrt((lat_obs - arr_lats)**2+((lon_obs-arr_lon)*np.cos(arr_lats*(np.pi/180)))**2)



lat, lon = [], []
df_match = pd.DataFrame()

for i in df_inv_stcamille.index.values: # lights in the inventoy
    
    lat_inv = float(df_inv_stcamille['lat_lights'].iloc[i])
    lon_inv = float(df_inv_stcamille['lon_lights'].iloc[i])
    
    list_dist = []
    for j in range(len(df_lan3_stcamille)): # lights detected from lan3
        lat_lights = float(df_lan3_stcamille['lat_lights'].iloc[j])
        lon_lights = float(df_lan3_stcamille['lon_lights'].iloc[j])

        list_dist.append(distance(lat_inv, lon_inv, lat_lights, lon_lights))
    
    arr_dist = np.array(list_dist)
    idx_dist = np.argwhere([list_dist < 0.00022])
    idx_dist = idx_dist[idx_dist != 0]
    
    if len(idx_dist) > 0:
        print(i)
        idx_dist = idx_dist[0]
        
        df = df_lan3_stcamille.iloc[[idx_dist]]
        df['H_detect'] = df_inv_stcamille.loc[i, 'H']
        df['tech_detect'] = df_inv_stcamille.loc[i, 'tech']
        
        lat.append(df_lan3_stcamille['lat_lights'].iloc[idx_dist])
        lon.append(df_lan3_stcamille['lon_lights'].iloc[idx_dist])
    
        df_match = pd.concat([df_match, df]) # Quoi faire si on a trouve plus que un index?
    
df_match.reset_index(drop=True, inplace=True)

# Validation détection position
print(f'Positions correctly detected:{len(df_match)}/{len(df_inv_stcamille)} ({round(len(df_match)/len(df_inv_stcamille)*100,2)}%)')


# Validation détection hauteur
H_relative = (df_match['H_detect'] / df_match['H']).values
good_H = H_relative[(H_relative > 0.5) & (H_relative < 1.5)]

good_tech = df_match[df_match['tech'] == df_match['tech_detect']]
print(f'Tech detection: {len(good_tech)/len(df_match)*100}')


# Write lat,lon to csv for validation
df_lat_lon = pd.DataFrame({'lat':lat,
                           'lon':lon})
df_lat_lon.to_csv('latlon_validation.csv', index=False)
