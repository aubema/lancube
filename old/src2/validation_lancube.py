# Script to validate to result of the Lancube
# Julien-Pierre Houle
# June 2022

import numpy as np
import pandas as pd

# Routiers
df_routier_stoke = pd.read_csv("Validation/validation_lancube - Stoke_routier.csv")
df_routier_st_cam = pd.read_csv("Validation/validation_lancube - St-Camille_routier.csv")
df_combine_routier = pd.concat([df_routier_stoke, df_routier_st_cam]).reset_index(drop=True)



def stats(df_inv, df_lancube):

    df_match = df_inv.dropna(subset=['lat_inv', 'lat_lan3']).reset_index(drop=True)
    nan_lan3 = df_inv['lat_lan3'].isnull().sum(axis=0)
    
    

    print(f"Nb Inventaire: {len(df_inv)}")
    print(f"Nb LanCube: {len(df_lancube)}")
    print(f"Nb match: {len(df_match)} - {len(df_match)/len(df_inv)*100:.2f}%")
    print(f"Fausse detection Lan3: {nan_lan3}")

    # Height
    # diff = ( (df_match['H_inv'] - df_match['H_lan3']) / df_match['H_inv']).values
    diff = (df_match['H_inv'] - df_match['H_lan3'])
    diff = diff[~np.isnan(diff)]

    diff_mean = np.mean(diff)
    diff_STD = np.std(diff)
    print(f'Diff mean : {diff_mean:.2f}')
    print(f'Diff STD : {diff_STD:.2f}')

    # Technologies
    same_tech = df_match[df_match['tech_inv'] == df_match['tech_lan3']]
    print(f"Tech accuracy: {len(same_tech)/len(df_match)*100:.3f}%")



# ROUTIERS STOKE
print('STOKE:')
lights_inv = df_routier_stoke.dropna(subset=['lat_inv']).reset_index(drop=True)
lights_lan3 =  df_routier_stoke.dropna(subset=['lat_lan3']).reset_index(drop=True)
stats(lights_inv, lights_lan3)



# ROUTIERS ST-CAMILLES
print('\n\nSt-Camille')
lights_inv = df_routier_st_cam.dropna(subset=['lat_inv']).reset_index(drop=True)
lights_lan3 =  df_routier_st_cam.dropna(subset=['lat_lan3']).reset_index(drop=True)
stats(lights_inv, lights_lan3)



# ROUTIERS COMBINÉS
print('\n\nCombiné')
df_inv = df_combine_routier.dropna(subset=['lat_inv'])
df_lancube =  df_combine_routier.dropna(subset=['lat_lan3'])
stats(df_inv, df_lancube)



df_inv = df_inv.dropna()
df_inv = df_inv[(df_inv['H_inv'] != 9.0) & (df_inv['H_inv'] != 12.0)]
corr = ((df_inv['H_inv'].mean()-2)/(df_inv['H_lan3'].mean()-2))**2
