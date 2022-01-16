#!/usr/bin/env python3

# Create discrete inventory from lancube data
# Author : Julien-Pierre Houle


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from src.cleaning_data import *
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)



PATH_DATAS = "Data/Mtl3_june_2021"
PATH_DATAS = "Data"

h = 2  # height of the lancube (m)
K = 1  # K factor


# Reading Lan3 datas
df = pd.read_csv(f"{PATH_DATAS}/Corr_2021-12-21.csv", sep=',', on_bad_lines='skip')
df['Value'] = df['Clear'] / df['Gain'] / df['AcquisitionTime(ms)']
# df['Value'] = df['lux']


# Correction temps acquision des couleurs
df['Red_i']   = df['Red']   / df['Gain'] / df['AcquisitionTime(ms)']
df['Green_i'] = df['Green'] / df['Gain'] / df['AcquisitionTime(ms)']
df['Blue_i']  = df['Blue']  / df['Gain'] / df['AcquisitionTime(ms)']
df['Clear']   = df['Clear'] / df['Gain'] / df['AcquisitionTime(ms)']

# Removing Infrared from RGB
df['Red']   = (df['Red_i']   - df['Blue_i'] - df['Green_i'] + df['Clear']) / 2
df['Green'] = (df['Green_i'] - df['Red_i']  - df['Blue_i']  + df['Clear']) / 2
df['Blue']  = (df['Blue_i']  - df['Red_i']  - df['Green_i'] + df['Clear']) / 2


# Cleaning dataframes
df1, df3, df5 = cleaning_data(df)

df35 = pd.DataFrame({ 'Value':   (df3['Value'].values - df5['Value'].values),
                      'Distance': df3['Traveled Distance'].values,
                      'R': abs(df3['Red'].values   - df5['Red'].values),
                      'G': abs(df3['Green'].values - df5['Green'].values),
                      'B': abs(df3['Blue'].values  - df5['Blue'].values) })

# Calculate Moving Average
df1['MA']  =  df1["Value"].rolling(window=3, center=True).mean()
df35['MA'] = df35['Value'].rolling(window=3, center=True).mean()

S1_value,  S1_dist  =  df1['Value'].values, df1['Traveled Distance'].values
S35_value, S35_dist =  df35['Value'].values, df35['Distance'].values



# ------------ Find index peaks S1 (top) & S3-S5 (side) ------------

# Note: For the sake of clarity and vigor, here is the notation used
# for the different vairables names:
# E  : Peaks (when a light fixture is detected)
# 1  : Sensor 1   (located on top of the lancube)
# 35 : Sensor 3-5 (located on the side of the lancube)
# S  : Simul (both top and side sendor detected a light)
# p  : prime (value just before or after the peaks)


print('Finding peaks and primes.')
idx_peak1,  _m  = find_peaks(df1['MA'],      height=0.04, prominence=0.02)
idx_peak35, _m = find_peaks(abs(df35['MA']), height=0.04, prominence=0.02)

# S1 & S35 peaks & traveled distance values
EV1  = S1_value[idx_peak1]
ED35 = df35['Distance'].iloc[idx_peak35]


idx_E1   = []
idx_ES1  = []
idx_ES35 = []

# Checking for side peaks close to top peak
for pos in idx_peak1:
    d = df1['Traveled Distance'].iloc[pos]  # traveled distance at peak (S1)
    
    mask_comb = ((d-3) < ED35) & (ED35 < (d+3)) # get simul peaks fom S1 and S35 (6m interval)
    comb = mask_comb[mask_comb]

    if len(comb) == 1:  # detection de un peak
        idx_ES1.append(pos)  # idx S1 peak
        idx_ES35.append(comb.index[0])  # idx S35 peak

    if len(comb) == 0:  # only top peak
        idx_E1.append(pos)

# Get index side lights (peak only in S3-S5)
idx_E35 = [i for i in idx_peak35 if i not in idx_ES35]




def find_prime(index_peaks, values, dist):
    """ Find prime value +- 2 points close to the index peak.
        Will take the back value if available, else the front one. """   

    idx = np.array(index_peaks, dtype=int)

    idxp_front = []
    idxp_back  = []
    
    mask_front = np.zeros(len(idx), dtype=bool)
    mask_back  = np.zeros(len(idx), dtype=bool)
    
    for j, i in enumerate(idx):
        
        idxp = np.argwhere( (dist > dist[i]+3) & (dist < dist[i]+20) ).flatten()
        idxp = idxp[ (abs(values[idxp]) < abs((values[i]*0.9)) ) ]
                
        if len(idxp > 0): # si on trouve un idx front
            idxp_front.append(idxp[0])
            mask_front[j] = True
            
        else:
            idxp = np.argwhere((dist > dist[i]-20) & (dist < dist[i]-3) ).flatten()  
            idxp = idxp[ abs(values[idxp]) < abs(values[i]*0.9) ]            
            
            if len(idxp > 0): # si on trouve un idx back
                idxp_back.append(idxp[-1])
                mask_back[j] = True
    
    idx_prime = np.concatenate([idxp_back, idxp_front]).astype(int)
    idx_prime.sort()
                    
    
    # Take front value if availaible else take back
    out = np.zeros_like(idx)
    out[mask_front] =  1
    out[mask_back]  = -1
    mask = out != 0
    idx_peaks = idx[mask]
    out = out[mask]
    
    return idx_prime, idx_peaks, out, mask


# Calling find prime function to get index
idx_ES1p, idx_ES1, out_ES1, deleting  = find_prime(idx_ES1, S1_value,  S1_dist)
idx_E1p,  idx_E1,  out_E1,  _del      = find_prime(idx_E1,  S1_value,  S1_dist)
idx_E35p, idx_E35, out_E35, _del      = find_prime(idx_E35, S35_value, S35_dist)


# ----- Get peaks and prime values -----
EVS1,  EVS1p  = S1_value[idx_ES1], S1_value[idx_ES1p]
EVS35 = df35['Value'].iloc[np.array(idx_ES35)[deleting]].values

EV1,  ED1  = S1_value[idx_E1],  S1_dist[idx_E1]
EV1p, ED1p = S1_value[idx_E1p], S1_dist[idx_E1p]

EV35,  ED35  = S35_value[idx_E35],  S35_dist[idx_E35]
EV35p, ED35p = S35_value[idx_E35p], S35_dist[idx_E35p]


# ----- Determining side of detection -----
side_simul = np.copy(EVS35).astype(str)
side_simul[side_simul < '0'] = 'left'
side_simul[side_simul != 'left'] = 'right'

side_S1 = np.empty(len(EV1))
side_S1[:] = np.nan

side_S35 = np.copy(EV35).astype(str)
side_S35[side_S35 < '0'] = 'left'
side_S35[side_S35 != 'left'] = 'right'



# ------------ Getting lights characteristics ------------
print('Getting lights characteristics.')

# Distance between peaks and prime
D_simul = abs(S1_dist[idx_ES1] - S1_dist[idx_ES1p])
D_S1    = abs(ED1  - ED1p)
D_S35   = abs(ED35 - ED35p)


# Horizontal distance between Lan3 and lights fixture
d_simul = D_simul / np.sqrt( ( (EVS1/EVS1p)**(2/3) * ( ((EVS1**2)/ (EVS35)**2)+1 ) \
             - ((EVS1**2)/ (EVS35)**2)-1) ) # (Eq. 14)
d_S1    = np.full(len(D_S1), 0)
d_S35   = D_S35 * abs(EV35p/EV35)**(1/3) / np.sqrt(1 - ( abs(EV35p/EV35)**(2/3) ) )  # (Eq. 27)


# Light fixture height.
H_simul = d_simul * (EVS1/abs(EVS35)) + h  # Eq. 15
H_S1    = (D_S1 * ((EV1p/EV1)**(1/3)) / np.sqrt(1 - ((EV1p/EV1)**(2/3))) ) + h  # (Eq. 21)
H_S35   = np.full(len(D_S35), h)  # lights with same hights as Lan3


# Line between light and lancube (Orthogonal) 
EOS1 = EVS1 * (np.sqrt((H_simul - h)**2 + d_simul**2)) / (H_simul - h)  # (Eq. 16)
EO1  = EV1
EO35 = abs(EV35)


# Concatenate scenarios data
D = np.concatenate([D_simul, D_S1, D_S35])
d = np.concatenate([d_simul, d_S1, d_S35])
H = np.concatenate([H_simul, H_S1, H_S35])
side = np.concatenate([side_simul, side_S1, side_S35])
EV   = np.concatenate([EVS1, EV1, EV35])
EO   = np.concatenate([EOS1, EO1, EO35])
out = np.concatenate([out_ES1, out_E1, out_E35])
# out = out * -1



# ----- Finding lights technologies (RGB) -----
 
idx_top_all = np.concatenate([idx_ES1, idx_E1])  # Top peaks idx from only S1 and simul S1
MRGB_top_all = np.stack((df1['Red'].iloc[idx_top_all].values,
                         df1['Green'].iloc[idx_top_all].values,
                         df1['Blue'].iloc[idx_top_all].values), axis=-1)

# Concatening S3 & S5 dataframe while maintening index order
MRGB_35 = np.stack( (df35['R'].iloc[idx_E35],
                     df35['G'].iloc[idx_E35],
                     df35['B'].iloc[idx_E35]), axis=-1 )

M_RGB = np.concatenate((MRGB_top_all, MRGB_35), axis=0) # Combine RGB arrays

MRB_G = np.vstack([ (M_RGB[:,0]/M_RGB[:,1]), (M_RGB[:,2]/M_RGB[:,1]) ]).T  # Columns R/G & B/G


# Reading lights datas
df_lights = pd.read_csv('Data/spectrum_colors.csv')
lights_RB_G = np.vstack([ df_lights['r/g_moy'], df_lights['b/g_moy'] ])

dist_color = np.sum( (MRB_G[:,:,None] - lights_RB_G)**2, 1)

idx_closest = np.argsort(dist_color)[:,:2]
closest_tech = df_lights['tech'].values[idx_closest]



MSI_tech = np.empty_like(closest_tech)
for i, tech in enumerate(closest_tech):
    MSI_tech[i] = [df_lights.loc[df_lights['tech'] == tech[0], 'MSI'].iloc[0],
                   df_lights.loc[df_lights['tech'] == tech[1], 'MSI'].iloc[0]]



# # Calcul du MSI pour les valeurs RGB
# MSI  = 0.0769 + 0.6023 * MRB_G[:, 1] - 0.1736 * MRB_G[:, 0] - 0.0489 * MRB_G[:,0] * MRB_G[:, 1] \
#      + 0.3098 * MRB_G[:, 1]**2 + 0.0257 * MRB_G[:, 0]**2

MSI = pd.concat([ df1['MSI'].iloc[idx_top_all],  df3['MSI'].iloc[idx_E35] ]).values


# Distance closest techs MSI & MSI peaks
dist_MSI = np.stack(( (MSI_tech[:,0] - MSI)**2,
                      (MSI_tech[:,1] - MSI)**2), axis=-1 )


# Keeping the tech with closest MSI
mask = (np.argmin(dist_MSI, axis=1) == 0)  

tech = closest_tech[ np.stack( (mask, ~mask), -1) ].reshape(-1, 1)


# Accessing ULOR in df base on technologie
ULOR = np.zeros(len(tech))
for i, techs in enumerate(tech):
    ULOR[i] = df_lights.loc[df_lights['tech'] == techs[0], 'ULOR'].iloc[0]

flux = (2*np.pi*K*EO) * (d**2 + (H-h)**2) / (1 - ULOR)




# ------------- Calculate lamps coordinate -------------
print('Determining lamps coordinate.')



# Peaks coords
lat_peak = pd.concat([ df1['Latitude'].iloc[idx_top_all],  df3['Latitude'].iloc[idx_E35]  ]).values
lon_peak = pd.concat([ df1['Longitude'].iloc[idx_top_all], df3['Longitude'].iloc[idx_E35] ]).values

# Prime coords
idxp_top = np.concatenate((idx_ES1p, idx_E1p))
lat_prime = pd.concat([ df1['Latitude'].iloc[idxp_top],  df3['Latitude'].iloc[idx_E35p]  ]).values
lon_prime = pd.concat([ df1['Longitude'].iloc[idxp_top], df3['Longitude'].iloc[idx_E35p] ]).values

# Get unit vector
delta_lon = (lon_prime - lon_peak) * out
delta_lat = (lat_prime - lat_peak) * out


vec_x = delta_lon * np.sin(lat_peak) /  \
        np.sqrt( (delta_lon*np.sin(lat_peak))**2  + (delta_lat)**2 )  # (eq 33)

vec_y = delta_lat / np.sqrt( (delta_lon*np.sin(lat_peak))**2  + (delta_lat)**2 ) # (eq 34)


epsilon = np.copy(side)
epsilon[epsilon == 'right'] = -np.pi/2
epsilon[epsilon == 'left']  =  np.pi/2
epsilon[epsilon == 'nan'] = 0
epsilon = epsilon.astype(float)


# Convert unit vector pointing the light fixture
vec_xl = (vec_x*np.cos(epsilon)) - (vec_y*np.sin(epsilon))  # (eq 35)
vec_yl = (vec_x*np.sin(epsilon)) + (vec_y*np.cos(epsilon))  # (eq 36)

# Light fixture position (meters)
xl = (d * vec_xl)  # (eq 37)
yl = (d * vec_yl)  # (eq 38)


lat_lights = ((180 * xl) / (np.pi*6373000)) + lat_peak # (eq 39)
lon_lights = ((180 * yl) / (np.pi*6373000*np.sin(lat_peak))) + lon_peak # (eq 49)




# ----------------- Writing data -----------------
print('Writing data.')

lux = df1['lux'].iloc[idx_top_all].tolist() + abs(df35['Value'].iloc[idx_E35]).tolist()
lux_MA = df1['MA'].iloc[idx_top_all].tolist() + abs(df35['MA'].iloc[idx_E35]).tolist()

df_invent = pd.DataFrame({
            'lat_lights': lat_lights,
            'lon_lights': lon_lights,
            'H'     : list(H),
            'tech'  : list(tech.flatten()),
            'lux'   : lux,
            'flux'  : flux,
            'side'  : list(side),
            'R/G'   : list(MRB_G[:,0]),
            'B/G'   : list(MRB_G[:,1]),
            'MSI'   : MSI,
            'd'     : list(d),
            'D'     : list(D),
            'lat_peaks' : lat_peak,
            'lon_peaks' : lon_peak,
            'E_perp': list(EO),
            })


df_invent['h'] = h
df_invent['Flag'] = np.where(df_invent['H']>20, 'Many', 'Single')


# df.to_csv('lan3_inventory.csv', index=False, sep=' ')
print('Done.')




# ************************************
#             Graphiques
# ************************************

plt.plot(df1['Traveled Distance'], df1['Value'], label='S1')
plt.plot(df35['Distance'], df35['Value'], label='S3-S5')

# # MA
# plt.plot(df1['Traveled Distance'], df1['MA'],  label='Top MA')
# plt.plot(df35['Distance'],         df35['MA'], label='Side MA')

# # PEAKS
# plt.plot(ED1,  EV1,  'o', c='red')
# plt.plot(ED1p, EV1p, 'o', c='pink') # Prime

plt.legend()
plt.show()