#!/usr/bin/env python3

# Create discrete inventory from lancube data
# Author : Julien-Pierre Houle


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from cleaning_data import *
import warnings
import datetime
warnings.filterwarnings("ignore",category=RuntimeWarning)


PATH_DATAS = "Data/St_camille"
filename = '2022-02-08.csv'
h = 2  # height of the lancube (m)
K = 1  # K factor


# Reading Lan3 datas
df = pd.read_csv(f"{PATH_DATAS}/{filename}", sep=',', on_bad_lines='skip')
df['Value'] = df['Clear'] / df['Gain'] / df['AcquisitionTime(ms)']


# Correction temps acquision des couleurs
df['Red_i']   = df['Red']   / df['Gain'] / df['AcquisitionTime(ms)']
df['Green_i'] = df['Green'] / df['Gain'] / df['AcquisitionTime(ms)']
df['Blue_i']  = df['Blue']  / df['Gain'] / df['AcquisitionTime(ms)']
df['Clear']   = df['Clear'] / df['Gain'] / df['AcquisitionTime(ms)']

# Removing Infrared from RGB
df['Red']   = (df['Red_i']   - df['Blue_i'] - df['Green_i'] + df['Clear']) / 2
df['Green'] = (df['Green_i'] - df['Red_i']  - df['Blue_i']  + df['Clear']) / 2
df['Blue']  = (df['Blue_i']  - df['Red_i']  - df['Green_i'] + df['Clear']) / 2

# Calculating IR
df['IR'] = (df['Red_i'] + df['Blue_i'] + df['Green_i'] - df['Clear']) / 2



# Cleaning dataframes
df1, df3, df5 = cleaning_data(df)

df35 = pd.DataFrame({ 'Value':   (df3['Value'].values - df5['Value'].values),
                      'Distance': df3['Traveled Distance'].values,
                      'R': abs(df3['Red'].values   - df5['Red'].values),
                      'G': abs(df3['Green'].values - df5['Green'].values),
                      'B': abs(df3['Blue'].values  - df5['Blue'].values),
                      'IR':abs(df3['IR'].values),
                      'Time': df3['Time'].values,
                      'lux' : (df3['lux'].values - df5['lux'].values) })

# Filtre Gaussien
df1['Value'] = gaussian_filter1d(df1['Value'].values, 0.4)
df35['Value'] = gaussian_filter1d(df35['Value'].values, 0.4)

df1['MA'] = gaussian_filter1d(df1['Value'].values, 0.4)
df35['MA'] = gaussian_filter1d(df35['Value'].values, 0.4)

S1_value,  S1_dist  =  df1['Value'].values,  df1['Traveled Distance'].values
S35_value, S35_dist =  df35['Value'].values, df35['Distance'].values

S1_value_MA,  S1_dist  =  df1['MA'].values,  df1['Traveled Distance'].values
S35_value_MA, S35_dist =  df35['MA'].values, df35['Distance'].values



# ------------ Find index peaks S1 (top) & S3-S5 (side) ------------

# Note: For the sake of clarity, here is the notation used
# for the different vairables names:
# E  : Peaks (when a light fixture is detected)
# 1  : Sensor 1   (located on top of the lancube)
# 35 : Sensor 3-5 (located on the side of the lancube)
# S  : Simul (both top and side sendor detected a light)
# p  : prime (value just before or after the peaks)

print('Finding peaks and primes.')
idx_peak1,  _m  = find_peaks(df1['MA'], distance=2, prominence=0.005, height=0.01)
idx_peak35, _m  = find_peaks(abs(df35['MA']), distance=2, prominence=0.005, height=0.01)


# # ------------------------------------------------
# plt.plot(df1['Traveled Distance'], df1['Value'], label='S1')
# plt.plot(df1['Traveled Distance'].iloc[idx_peak1], df1['Value'].iloc[idx_peak1],  'o', c='red')
# plt.show()
# # ------------------------------------------------

# S1 & S35 peaks & traveled distance values
EV1  = S1_value[idx_peak1]
ED35 = df35['Distance'].iloc[idx_peak35]


# Checking for side peaks close to top peak
idx_E1, idx_ES1, idx_ES35 = [], [], []

for pos in idx_peak1:
    
    d = df1['Traveled Distance'].iloc[pos]  # traveled distance at peak (S1)
    
    mask_comb = ((d-5) < ED35) & (ED35 < (d+5)) # get simul peaks fom S1 and S35 (6m interval)
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
        
        idxp = np.argwhere( (dist > dist[i]+3) & (dist < dist[i]+30) ).flatten()
        idxp = idxp[abs(values[idxp]) < abs(values[i]*0.90)] # minimal decrease of 10%
        idxp = idxp[abs(values[idxp]) > abs(values[i]*0.40)] # maximal decrease of 70%   
        idxp = idxp[values[idxp]/values[i] > 0] # make sure to have the same sign for both
       
                
        if len(idxp > 0): # si on trouve un idx front
            idxp_front.append(idxp[0])
            mask_front[j] = True
            
        else:
            idxp = np.argwhere((dist > dist[i]-30) & (dist < dist[i]-3) ).flatten()  
            idxp = idxp[abs(values[idxp]) < abs(values[i]*0.90)] # minimal decrease of 10%     
            idxp = idxp[abs(values[idxp]) > abs(values[i]*0.40)] # maximal decrease of 70%          
            idxp = idxp[values[idxp]/values[i] > 0] # make sure to have the same sign for both     
            
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
idx_ES1p, idx_ES1, out_ES1, deleting = find_prime(idx_ES1, S1_value,  S1_dist)
idx_E1p,  idx_E1,  out_E1,  _del     = find_prime(idx_E1,  S1_value,  S1_dist)
idx_E35p, idx_E35, out_E35, _del     = find_prime(idx_E35, S35_value, S35_dist)


# ------------------------------------------------
plt.plot(df1['Traveled Distance'], df1['Value'], label='S1')
plt.plot(df1['Traveled Distance'], df1['MA'],    label='MA')

plt.plot(df1['Traveled Distance'].iloc[idx_E1], df1['Value'].iloc[idx_E1],  'o', c='red')
plt.plot(df1['Traveled Distance'].iloc[idx_ES1], df1['Value'].iloc[idx_ES1],  'o', c='green')
plt.legend()
plt.show()
# ------------------------------------------------



# ----- Get peaks and prime values -----

EVS1,  EVS1p  = S1_value[idx_ES1], S1_value[idx_ES1p]
EDS1,  EDS1p  = S1_dist[idx_ES1],  S1_dist[idx_ES1p]
EVS35 = df35['Value'].iloc[np.array(idx_ES35)[deleting]].values  # remove values with no prime from S1

EV1,  ED1  = S1_value[idx_E1],  S1_dist[idx_E1]
EV1p, ED1p = S1_value[idx_E1p], S1_dist[idx_E1p]

EV35,  ED35  = S35_value[idx_E35],  S35_dist[idx_E35]
EV35p, ED35p = S35_value[idx_E35p], S35_dist[idx_E35p]


# Apply filter to make sure prime are smaller than peaks
mask_peaks_E35 = EV35p/EV35 < 1
mask_peaks_ES1 = EVS1p/EVS1 < 1
mask_peaks_E1  = EV1p/EV1 < 1

EV35, ED35 = EV35[mask_peaks_E35], ED35[mask_peaks_E35]
EV35p, ED35p = EV35p[mask_peaks_E35], ED35p[mask_peaks_E35]

EV1,  ED1  = EV1[mask_peaks_E1],  ED1[mask_peaks_E1]
EV1p, ED1p = EV1p[mask_peaks_E1], ED1p[mask_peaks_E1]

EVS1,  EVS1p  = EVS1[mask_peaks_ES1],  EVS1p[mask_peaks_ES1]
EDS1,  EDS1p  = EDS1[mask_peaks_ES1],  EDS1p[mask_peaks_ES1]
EVS35 = EVS35[mask_peaks_ES1]

idx_ES1p, idx_ES1, out_ES1 = idx_ES1p[mask_peaks_ES1], idx_ES1[mask_peaks_ES1], out_ES1[mask_peaks_ES1]
idx_E1p,  idx_E1,  out_E1  = idx_E1p[mask_peaks_E1],   idx_E1[mask_peaks_E1],   out_E1[mask_peaks_E1]
idx_E35p, idx_E35, out_E35 = idx_E35p[mask_peaks_E35], idx_E35[mask_peaks_E35], out_E35[mask_peaks_E35]


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


# ----- Finding lights technologies (RGB) -----
 
idx_top_all = np.concatenate([idx_ES1, idx_E1])  # Top peaks idx from only S1 and simul S1
MRGB_top_all = np.stack((df1['Red'].iloc[idx_top_all].values,
                         df1['Green'].iloc[idx_top_all].values,
                         df1['Blue'].iloc[idx_top_all].values,
                         df1['IR'].iloc[idx_top_all]), axis=-1)

# Concatening S3 & S5 dataframe while maintening index order
MRGB_35 = np.stack( (df35['R'].iloc[idx_E35],
                     df35['G'].iloc[idx_E35],
                     df35['B'].iloc[idx_E35],
                     df35['IR'].iloc[idx_E35]), axis=-1 )

M_RGBI = np.concatenate((MRGB_top_all, MRGB_35), axis=0) # Combine RGB arrays

MRBI_G = np.vstack([ (M_RGBI[:,0]/M_RGBI[:,1])*0.14, # R/G
                     (M_RGBI[:,2]/M_RGBI[:,1]), # B/G
                     (M_RGBI[:,3]/M_RGBI[:,1])]).T  # I/G


# # ------------------------------------------------
# plt.plot(df1['Traveled Distance'], df1['Value'], label='S1')
# plt.plot(df1['Traveled Distance'].iloc[idx_top_all], df1['Value'].iloc[idx_top_all], 'o', c='red')
# plt.show()
# # ------------------------------------------------

# -------  Reading lights datas -------
df_lights = pd.read_csv('Data/spectrum_colors.csv')

lights_RBI_G = np.vstack([df_lights['r/g']*0.14, df_lights['b/g'], df_lights['i/g']])
dist_color = np.sum((MRBI_G[:,:,None] - lights_RBI_G)**2, 1)

idx_closest = np.argsort(dist_color)[:,:1]
closest_tech = df_lights['tech'].values[idx_closest]


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
H_simul = d_simul * (EVS1/abs(EVS35)) + h  # (Eq. 15)
H_S1    = (D_S1 * ((EV1p/EV1)**(1/3)) / np.sqrt(1 - ((EV1p/EV1)**(2/3))) ) + h  # (Eq. 21)
H_S35   = np.full(len(D_S35), h)  # lights with same hights as Lan3


# Line between light and lancube (Orthogonal) 
EOS1 = df1['lux'].iloc[idx_ES1].values * (np.sqrt((H_simul - h)**2 + d_simul**2)) / (H_simul - h)  # (Eq. 16)
EO1  = df1['lux'].iloc[idx_E1].values
EO35 = abs(df35['lux'].iloc[idx_E35].values)


# Concatenate scenarios data
D = np.concatenate([D_simul, D_S1, D_S35])
d = np.concatenate([d_simul, d_S1, d_S35])
H = np.concatenate([H_simul, H_S1, H_S35])
side = np.concatenate([side_simul, side_S1, side_S35])
EV   = np.concatenate([EVS1, EV1, EV35])
EO   = np.concatenate([EOS1, EO1, EO35])
out  = np.concatenate([out_ES1, out_E1, out_E35])


# Accessing corrresponding ULOR
ULOR = np.zeros(len(closest_tech))
for i, techs in enumerate(closest_tech):
    ULOR[i] = df_lights.loc[df_lights['tech'] == techs[0], 'ULOR'].iloc[0]

flux = (2*np.pi*K*EO) * (d**2 + (H-h)**2) / (1 - ULOR)



# ------------- Calculate lamps coordinate -------------
print('Determining lamps coordinate.')

# Peaks coords
lat_peak = pd.concat([ df1['Latitude'].iloc[idx_top_all],  df3['Latitude'].iloc[idx_E35]  ]).values
lon_peak = pd.concat([ df1['Longitude'].iloc[idx_top_all], df3['Longitude'].iloc[idx_E35] ]).values



# Prime coords
idxp_top = np.concatenate((idx_ES1p, idx_E1p))
lat_prime = pd.concat([ df1['Latitude'].iloc[idxp_top],  df3['Latitude'].iloc[idx_E35p] ]).values
lon_prime = pd.concat([ df1['Longitude'].iloc[idxp_top], df3['Longitude'].iloc[idx_E35p] ]).values

# Get unit vector
delta_lon = (lon_prime - lon_peak) * out
delta_lat = (lat_prime - lat_peak) * out

vec_x = delta_lon * np.sin(lat_peak) /  \
        np.sqrt( (delta_lon*np.sin(lat_peak))**2  + (delta_lat)**2 )  # (eq 33)

vec_y = delta_lat / np.sqrt( (delta_lon*np.sin(lat_peak))**2  + (delta_lat)**2 ) # (eq 34)

epsilon = np.copy(side)
epsilon[epsilon == 'right'] = -np.pi/2
epsilon[epsilon == 'left']  = np.pi/2
epsilon[epsilon == 'nan'] = 0
epsilon = epsilon.astype(float)


# Convert unit vector pointing the light fixture
vec_xl = (-vec_y*np.sin(epsilon))  # (eq 35)
vec_yl = (vec_x*np.sin(epsilon))  # (eq 36)

# Light fixture position (meters)
xl = (d * vec_xl)  # (eq 37)
yl = (d * vec_yl)  # (eq 38)

lat_lights = ((180 * yl) / (np.pi*6373000)) + lat_peak # (eq 39)
lon_lights = ((180 * xl) / (np.pi*6373000*np.sin(lat_peak))) + lon_peak # (eq 40)








# ----------------- Writing data -----------------
print('Writing data.')

lux = df1['lux'].iloc[idx_top_all].tolist() + abs(df35['Value'].iloc[idx_E35]).tolist()
lux_MA = df1['MA'].iloc[idx_top_all].tolist() + abs(df35['MA'].iloc[idx_E35]).tolist()
time = df1['Time'].iloc[idx_top_all].tolist() + df35['Time'].iloc[idx_E35].tolist()


df_invent = pd.DataFrame({
            'lat_lights': lat_lights,
            'lon_lights': lon_lights,
            'H'     : list(H),
            'tech'  : list(closest_tech.flatten()),
            'lux'   : lux,
            'flux'  : flux,
            'side'  : list(side),
            'R/G'   : list(MRBI_G[:,0]),
            'B/G'   : list(MRBI_G[:,1]),
            'd'     : list(d),
            'D'     : list(D),
            'lat_peaks' : lat_peak,
            'lon_peaks' : lon_peak,
            'E_perp': list(EO),
            'time'  : time,
            'out' : out
            })

df_invent['h'] = h
df_invent['Flag'] = np.where(df_invent['H']>20, 'Many', 'Single')
df_invent.to_csv(f'inventaires/lan3_invent_{filename}', index=False)
print('Done.')



# ************************************
#             Graphiques
# ************************************

# plt.plot(df1['Traveled Distance'], df1['Value'], label='S1')
# plt.plot(df35['Distance'], df35['Value'], label='S3-S5')

ED1_all = np.concatenate([ED1, EDS1])
EV1_all = np.concatenate([EV1, EVS1])

ED1p_all = np.concatenate([ED1p, EDS1p])
EV1p_all = np.concatenate([EV1p, EVS1p])


plt.plot(df1['Traveled Distance'], df1['Value'], label='S1')
plt.plot(ED1_all,  EV1_all,  'o', c='red')  # Peaks
plt.plot(ED1p_all, EV1p_all, 'o', c='pink') # Prime

plt.plot(df35['Distance'], df35['Value'], label='S35')
plt.plot(ED35, EV35, 'o', c='red') # Peaks
plt.plot(ED35p, EV35p, 'o', c='green') # Prime

plt.legend()
plt.show()
