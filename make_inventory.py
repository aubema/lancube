#!/usr/bin/env python3

# Make discrete inventory from lancube data
# Author : Julien-Pierre Houle
# Last update : August 2021


import os
import numpy as np
import pandas as pd
import osmnx as ox
import progressbar
from datetime import datetime
from pyproj import CRS, Transformer
import warnings
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage.interpolation import shift
from cleaning_data import *

warnings.filterwarnings('ignore') # ignore numpy warning

# PARAMETERS
PATH_DATAS = "Data/Mtl_june_2021"
light_folder = 'Lights_cloud/' # path to the lights cloud folder
street_distance = 5 # Maximal distance(m) between coord and street
h = 2   # height of the lancube in meters
K = 1  # K factor



# Load Lights tech clouds
print('Loading lights tech cloud.')
list_tech = []
frames = []

for filename in os.listdir(light_folder):
    df_cloud = pd.read_csv(f"{light_folder}/{filename}")
    list_tech += [filename[:-4]] * len(df_cloud)
    frames.append(df_cloud)

df_lights = pd.concat(frames, ignore_index=True)
lights_cloud = np.array((df_lights['RC_avg'].values,
                         df_lights['GC_avg'].values,
                         df_lights['BC_avg'].values))



df = pd.read_csv(f"{PATH_DATAS}/2021-06-07.csv", sep=',', on_bad_lines='skip')

# Caluculate Value
df['Value'] = df['Clear'] / df['Gain'] / df['Acquisition time (ms)']

# Cleaning dataframes
df1, df3, df5 = cleaning_data(df)

S1_value = df1['Value'].values
S1_dist  = df1['Traveled Distance'].values

df35 = pd.DataFrame({'Value':    (df3['Value'].values - df5['Value'].values),
                    'Distance': df3['Traveled Distance'].values,
                    'BC_3': (df3['Blue']  / df3['Clear']).values,
                    'RC_3': (df3['Red']   / df3['Clear']).values,
                    'GC_3': (df3['Green'] / df3['Clear']).values,
                    'BC_5': (df5['Blue']  / df5['Clear']).values,
                    'RC_5': (df5['Red']   / df5['Clear']).values,
                    'GC_5': (df5['Green'] / df5['Clear']).values})

# Calculate Moving Average
df1['MA'] = df1["Value"].rolling(window=3, center=True).mean()
df35['MA'] = df35['Value'].rolling(window=3, center=True).mean()



# ------------ Find index peaks S1 & S3-S5 (side) ------------

print('\nFinding peaks and primes.')
idx_EV1, _m = find_peaks(df1['MA'],      height=0.04, prominence=0.02)
idx_EVS, _m = find_peaks(abs(df35['MA']), height=0.04, prominence=0.02)

# S1 & S35 peaks & distance values
EV1, ED1 = S1_value[idx_EV1], S1_dist[idx_EV1]
EV35, ED35 = df35['Value'].iloc[idx_EVS], df35['Distance'].iloc[idx_EVS].values


idx_simul = []
idx_simul_side = []
idx_S1  = []

# Check for (S3-S5) peak close to S1 peak
for pos in idx_EV1:
    d = S1_dist[pos] # distance peak S1

    simul_dist = ED35[((d-3) < ED35) & (ED35 < (d+3))] # ED35 in a 6m interval from S1 peak

    if len(simul_dist) == 1:  # high light fixture
        idx_simul.append(pos)
        idx_simul_side.append(df35['Distance'].tolist().index(simul_dist[0]))

    if len(simul_dist) == 0: # peak only in S1
        idx_S1.append(pos)


# Get index low lights (peak only in S3-S5)
idx_S35 = [elem for elem in idx_EVS if elem not in idx_simul_side]



def find_prime(index_peaks, values, dist):
    """ Find prime value +- 2 points close to the index peak.
        Will take the left value if available else the right one. """

    idx = np.array(index_peaks)
    mask_L = (abs(values[idx-2]) < abs(values[idx])) & (dist[idx]-dist[idx-2] < 15)
    mask_R = (abs(values[idx+2]) < abs(values[idx])) & (dist[idx+2]-dist[idx] < 15)
    idx_left  = np.where(mask_L, idx, np.nan) -2
    idx_right = np.where(mask_R, idx, np.nan) +2

    # Take left value if availaible else take right
    nan_min = np.nanmin([idx_left, idx_right], axis=0)
    mask = ~np.isnan(nan_min)

    idx_prime = nan_min[mask].astype(int)
    idx_peaks = np.array(index_peaks)[mask].astype(int)

    return idx_prime, idx_peaks, mask

# Calling find prime function to get index
prime_simul, peaks_simul, deleting = find_prime(idx_simul, S1_value, S1_dist)
prime_S1,    peaks_S1,    _del     = find_prime(idx_S1, S1_value, S1_dist)
prime_S35,   peaks_S35,   _del     = find_prime(idx_S35, df35['Value'].values, df35['Distance'].values)


# Get peaks and prime values
EV, ED = S1_value[peaks_simul], S1_dist[peaks_simul]
EVp, EDp = S1_value[prime_simul], S1_dist[prime_simul]
EV_side  = df35['Value'].iloc[np.array(idx_simul_side)[deleting]].values # S35 peaks when simul

EV1, ED1 = S1_value[peaks_S1], S1_dist[peaks_S1]
EV1p, ED1p = S1_value[prime_S1], S1_dist[prime_S1]

EV35, ED35 = df35['Value'].iloc[peaks_S35].values, df35['Distance'].iloc[peaks_S35].values
EV35p, ED35p = df35['Value'].iloc[prime_S35].values, df35['Distance'].iloc[prime_S35].values


# Determining side
side_simul = np.copy(EV_side).astype(str)
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
D_simul = abs(ED - EDp)
D_S1    = abs(ED1 - ED1p)
D_S35   = abs(ED35 - ED35p)

# Horizontal distance between Lan3 and lights fixture at t.
d_simul = D_simul / np.sqrt( ( (EV/EVp)**(2/3) * ( ((EV**2)/ (EV_side)**2)+1 ) - ((EV**2)/ (EV_side)**2)-1) )
d_S1    = np.full(len(D_S1), 0)
d_S35   = D_S35 * abs(EV35p/EV35)**(1/3) / np.sqrt(1 - ( abs(EV35p/EV35) **(2/3)) )

# Light fixture height.
H_simul = d_simul * (EV/abs(EV_side)) + h
H_S1    =  (D_S1 * ((EV1p/EV1)**(1/3)) / np.sqrt(1 - ((EV1p/EV1)**(2/3))) ) + h
H_S35   = np.full(len(D_S35), h)


# Equation 16
Eperp_simul = EV * (np.sqrt((H_simul - h)**2 + d_simul**2)) / (H_simul - h)
Eperp_S1    = EV1
Eperp_S35   = abs(EV35)


# Concatenate scenarios data
D = np.concatenate([D_simul, D_S1, D_S35])
d = np.concatenate([d_simul, d_S1, d_S35])
H = np.concatenate([H_simul, H_S1, H_S35])
side = np.concatenate([side_simul, side_S1, side_S35])
E_val  = np.concatenate([EV, EV1, EV35])
E_perp = np.concatenate([Eperp_simul, Eperp_S1, Eperp_S35])


# Find nearest point in the lights technologie cloud
peak_simul_S1 = np.concatenate([peaks_simul, peaks_S1])
M_simul_S1 = np.stack( ((df['Blue']  / df['Clear']).iloc[peak_simul_S1].values,
                        (df['Green'] / df['Clear']).iloc[peak_simul_S1].values,
                        (df['Red']   / df['Clear']).iloc[peak_simul_S1].values), axis=-1)


S35_peaks_val = df35['Value'].iloc[peaks_S35]
idx_S3 = S35_peaks_val[S35_peaks_val > 0].index.values
idx_S5 = S35_peaks_val[S35_peaks_val < 0].index.values

# Convatening both dataframe (S3 & S5) while maintening index order
M_S35 = np.stack((
        pd.concat([df35['BC_3'].iloc[idx_S3], df35['BC_5'].iloc[idx_S5]], sort=False).sort_index().values,
        pd.concat([df35['GC_3'].iloc[idx_S3], df35['GC_5'].iloc[idx_S5]], sort=False).sort_index().values,
        pd.concat([df35['RC_3'].iloc[idx_S3], df35['RC_5'].iloc[idx_S5]], sort=False).sort_index().values),
        axis=-1)

# Combine the 2D arrays
M = np.concatenate((M_simul_S1, M_S35), axis=0)

distance = np.sum((M[:,:,None] - lights_cloud) **2, 1)
idx_closest = np.argmin(distance, 1)
lights_tech  = np.array(list_tech)[idx_closest] # closest light technologie


# Estimating the light flux
uplight_dict = {'HPS': 0.5,
                'MH' : 0.5,
                'LED': 0.0}

U = np.array([uplight_dict[t] for t in lights_tech])

flux = (2*np.pi*K*E_perp) * (d**2 + (H-h)**2) / (1 - U)



# ----------------- Writing data -----------------
print('Writing data.')

lats = df1['Latitude'].iloc[peak_simul_S1].tolist()  + df3['Latitude'].iloc[peaks_S35].tolist()
lons = df1['Longitude'].iloc[peak_simul_S1].tolist() + df3['Longitude'].iloc[peaks_S35].tolist()

M_RGB = np.hsplit(M, 3)

df_invent = pd.DataFrame({
            'lat'   : lats,
            'lon'   : lons,
            'D'     : list(D),
            'd'     : list(d),
            'H'     : list(H),
            'tech'  : list(lights_tech),
            'side'  : list(side),
            'E_perp': list(E_perp),
            'flux'  : list(flux),
            'R/C'   : list(M_RGB[0].flatten()),
            'G/C'   : list(M_RGB[1].flatten()),
            'B/C'   : list(M_RGB[2].flatten())
            })

df_invent['h'] = h
# df.to_csv('lan3_inventory.csv', index=False, sep=' ')
print('\nDone.')


# ********************************************************************

plt.plot(df1['Traveled Distance'], df1['Value'], label='S1')
plt.plot(df35['Distance'], df35['Value'], label='S3-S5')

# MA
plt.plot(df1['Traveled Distance'], df1['MA'], label='S1 MA')
plt.plot(df35['Distance'], df35['MA'], label='Side MA')

# PEAKS
plt.plot(ED1, EV1, 'o', c='red')
plt.plot(EDp, EVp, 'o', c='pink') # Prime

plt.legend()
plt.show()
