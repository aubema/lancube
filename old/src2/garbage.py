#!/usr/bin/env python3

# Create discrete inventory from lancube data
# Author : Julien-Pierre Houle


from curses import window
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from src.cleaning_data import *
from src.gaussian_filter import *
import warnings
import datetime
warnings.filterwarnings("ignore",category=RuntimeWarning)
pd.options.mode.chained_assignment = None


# PARAMETERS
PATH_DATAS = "Data/St_camille"
filename = '2022-02-07-08.csv'
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

# df35 = pd.DataFrame({ 'Value':   (df3['Value'].values - df5['Value'].values),
#                       'Distance': df3['Traveled Distance'].values,
#                       'R': abs(df3['Red'].values   - df5['Red'].values),
#                       'G': abs(df3['Green'].values - df5['Green'].values),
#                       'B': abs(df3['Blue'].values  - df5['Blue'].values),
#                       'IR':abs(df3['IR'].values),
#                       'Time': df3['Time'].values,
#                       'lux' : (df3['lux'].values - df5['lux'].values) })

# Filtre Gaussien
df1['Value'] = gaussian_filter1d(df1['Value'].values, 0.4)
df3['Value'] = gaussian_filter1d(df3['Value'].values, 0.4)
df5['Value'] = gaussian_filter1d(df5['Value'].values, 0.4)

# Remove background contribution
df1['Value'] -= df1['Value'].rolling(window=5).min()
df3['Value'] -= df3['Value'].rolling(window=5).min()
df5['Value'] -= df5['Value'].rolling(window=5).min()

# # Plot graphique
# plt.figure()
# plt.plot( df1['Traveled Distance'], df1['Value'] , label='S1')
# plt.plot( df3['Traveled Distance'], df3['Value'] , label='S3')
# plt.plot( df5['Traveled Distance'], df5['Value'] , label='S5')
# plt.show()


S1_value, S1_dist = df1['Value'].values, df1['Traveled Distance'].values
S3_value, S3_dist = df3['Value'].values, df3['Traveled Distance'].values
S5_value, S5_dist = df5['Value'].values, df5['Traveled Distance'].values


# ------------ Find index peaks S1 (top) & S3, S5 (side) ------------

# Note: For the sake of clarity, here is the notation used
# for the different vairables names:
# E  : Peaks (when a light fixture is detected)
# 1  : Sensor 1   (located on top of the lancube)
# 35 : Sensor 3-5 (located on the side of the lancube)
# S  : Simul (both top and side sendor detected a light)
# p  : prime (value just before or after the peaks)

print('Finding peaks and primes.')
idx_peak1,  _m  = find_peaks(df1['Value'], distance=2, prominence=0.005, height=0.01)
idx_peak3,  _m  = find_peaks(df3['Value'], distance=2, prominence=0.005, height=0.01)
idx_peak5,  _m  = find_peaks(df5['Value'], distance=2, prominence=0.005, height=0.01)


# Get peaks & traveled distance values
EV1, EV3, EV5 = S1_value[idx_peak1], S3_value[idx_peak3], S5_value[idx_peak5]
ED1, ED3, ED5 = S1_dist[idx_peak1], S3_dist[idx_peak3], S1_dist[idx_peak5]

# Checking for side peaks close to top peak
idx_E1, idx_ES1, idx_ES3, idx_ES5 = [], [], [], []

for pos in idx_peak1:
    
    d = df1['Traveled Distance'].iloc[pos]  # traveled distance at peak (S1)
    
    mask_comb3 = ((d-5) < ED3) & (ED3 < (d+5)) # get simul peaks fom S1 and S33 (10m interval)
    mask_comb5 = ((d-5) < ED5) & (ED5 < (d+5)) # get simul peaks fom S1 and S33 (10m interval)
    comb3 = mask_comb3[mask_comb3]
    comb5 = mask_comb5[mask_comb5]

    if len(comb3) == 1:  # detection de un peak
        idx_ES1.append(pos)  # idx S1 peak
        idx_ES3.append(comb3.index[0])  # idx S35 peak
        
    # else:
    #     idx_E1
        
    if len(comb5) == 1:
        idx_ES1.append(pos)  # idx S1 peak
        idx_ES5.append(comb5.index[0])  # idx S35 peak

    if len(comb3) + len(comb5) == 0:  # only top peak
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
idx_ES3p, idx_ES3, out_ES3, deleting = find_prime(idx_ES3, S1_value,  S1_dist)
idx_ES5p, idx_ES5, out_ES5, deleting = find_prime(idx_ES5, S1_value,  S1_dist)
idx_E1p,  idx_E1,  out_E1,  _del     = find_prime(idx_E1,  S1_value,  S1_dist)
idx_E3p, idx_E3, out_E3,    _del     = find_prime(idx_E3,  S5_value,  S5_dist)
idx_E5p, idx_E5, out_E5,    _del     = find_prime(idx_E5,  S5_value,  S5_dist)


# # ------------------------------------------------
# plt.plot(df1['Traveled Distance'], df1['Value'], label='S1')
# plt.plot(df1['Traveled Distance'].iloc[idx_E1], df1['Value'].iloc[idx_E1],  'o', c='red')
# plt.plot(df1['Traveled Distance'].iloc[idx_ES1], df1['Value'].iloc[idx_ES1],  'o', c='green')
# plt.legend()
# plt.show()
# # ------------------------------------------------



# ----- Get peaks and prime values -----

EVS3,  EVS3p  = S1_value[idx_ES3], S1_value[idx_ES3p]
EVS5,  EVS5p  = S1_value[idx_ES5], S1_value[idx_ES5p]

EDS3,  EDS3p  = S1_dist[idx_ES3],  S1_dist[idx_ES3p]
EDS5,  EDS5p  = S1_dist[idx_ES3],  S1_dist[idx_ES5p]

# EVS35 = df3['Value'].iloc[np.array(idx_ES3)[deleting]].values #remove values with no prime from S1

EV1,  ED1  = S1_value[idx_E1],  S1_dist[idx_E1]
EV1p, ED1p = S1_value[idx_E1p], S1_dist[idx_E1p]

EV3,  ED3 = S3_value[idx_E3],  S3_dist[idx_E3]
EV5,  ED5 = S5_value[idx_E5],  S5_dist[idx_E5]
EV3p, ED3p = S3_value[idx_E3p], S3_dist[idx_E3p]
EV5p, ED5p = S5_value[idx_E5p], S5_dist[idx_E5p]

# # Apply filter to make sure prime are smaller than peaks
# mask_peaks_E3 = EV3p/EV3 < 1
# mask_peaks_E5 = EV5p/EV5 < 1
# mask_peaks_ES3 = EVS3p/EVS3 < 1
# mask_peaks_ES5 = EVS5p/EVS5 < 1
# mask_peaks_E1  = EV1p/EV1 < 1
# EV1,  ED1  = EV1[EV1p/EV1 < 1],  ED1[EV1p/EV1 < 1]
# EV1p, ED1p = EV1p[EV1p/EV1 < 1], ED1p[EV1p/EV1 < 1]
# EV3, ED3 = EV3[EV3p/EV3 < 1], ED3[EV3p/EV3 < 1]
# EV3p, ED3p = EV3p[EV3p/EV3 < 1], ED1p[EV3p/EV3 < 1]
# EV5, ED5 = EV5[EV5p/EV5 < 1], ED5[EV5p/EV5 < 1]
# EV5p, ED5p = EV5p[EV5p/EV5 < 1], ED5p[EV5p/EV5 < 1]
# EVS3, EDS3 = EVS3[EVS3p/EVS3 < 1], EDS3[EVS3p/EVS3 < 1]
# EVS3p, EDS3p = EVS3p[EVS3p/EVS3 < 1], EDS3p[EVS3p/EVS3 < 1]
# EVS5, EDS5 = EVS5[EVS5p/EVS5 < 1], EDS5[EVS5p/EVS5 < 1]
# EVS5p, EDS5p = EVS5p[EVS5p/EVS5 < 1], EDS5p[EVS5p/EVS5 < 1]
# idx_E1p,  idx_E1,  out_E1  = idx_E1p[mask_peaks_E1],   idx_E1[mask_peaks_E1],   out_E1[mask_peaks_E1]
# idx_ES3p, idx_ES3,out_ES3 = idx_ES1p[EVS3p/EVS3 < 1],idx_ES1[EVS3p/EVS3 < 1], out_ES1[EVS3p/EVS3 < 1]
# idx_E35p, idx_E35, out_E35 = idx_E35p[mask_peaks_E35], idx_E35[mask_peaks_E35], out_E35[mask_peaks_E35]


# ----- Determining side of detection -----
side_1 = np.full(len(EV1, np.nan))
side_3 = np.full(len(EV3, 'right'))
side_5 = np.full(len(EV5, 'left'))
side_S3 = np.full(len(EVS3, 'right'))
side_S5 = np.full(len(EVS5, 'left'))


# ------------ Getting lights characteristics ------------
print('Getting lights characteristics.')


# ----- Finding lights technologies (RGB) ----- 
idx_top_all = np.concatenate([idx_E1, idx_ES3, idx_ES5])  # Top peaks idx from only S1 and simul S1
MRGB_top_all = np.stack(( df1['Red'].iloc[idx_top_all].values,
                          df1['Green'].iloc[idx_top_all].values,
                          df1['Blue'].iloc[idx_top_all].values,
                          df1['IR'].iloc[idx_top_all]), axis=-1 )

# Concatening S3 & S5 dataframe while maintening index order
MRGB_3 = np.stack( (df3['R'].iloc[idx_E3],
                     df3['G'].iloc[idx_E3],
                     df3['B'].iloc[idx_E3],
                     df3['IR'].iloc[idx_E3]), axis=-1 )

MRGB_5 = np.stack( (df5['R'].iloc[idx_E5],
                     df5['G'].iloc[idx_E5],
                     df5['B'].iloc[idx_E5],
                     df5['IR'].iloc[idx_E5]), axis=-1 )

M_RGBI = np.concatenate((MRGB_top_all, MRGB_3, MRGB_5), axis=0) # Combine RGB arrays

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
D_S1    = abs(ED1  - ED1p)
D_S3   = abs(ED3 - ED3p)
D_S5   = abs(ED5 - ED5p)
D_simul3 = abs(S1_dist[idx_ES3] - S1_dist[idx_ES3p])
D_simul5 = abs(S1_dist[idx_ES5] - S1_dist[idx_ES5p])

        
# Horizontal distance between Lan3 and lights fixture
d_S1    = np.full(len(D_S1), 0)
d_S3   = D_S3 * abs(EV3p/EV3)**(1/3) / np.sqrt(1 -( abs(EV3p/EV3)**(2/3) ) )  # (Eq. 27)
d_S5   = D_S5 * abs(EV5p/EV5)**(1/3) / np.sqrt(1 -( abs(EV5p/EV5)**(2/3) ) )  # (Eq. 27)

d_simul3 = D_simul3 / np.sqrt( ( (EVS3/EVS3p)**(2/3) * ( ((EVS3**2)/ (EVS35)**2)+1 ) \
            - ((EVS3**2)/ (EVS35)**2)-1) ) # (Eq. 14)
d_simul5 = D_simul5 / np.sqrt( ( (EVS5/EVS5p)**(2/3) * ( ((EVS5**2)/ (EVS35)**2)+1 ) \
            - ((EVS5**2)/ (EVS35)**2)-1) ) # (Eq. 14)


# Light fixture height.
H_S1    = (D_S1 * ((EV1p/EV1)**(1/3)) / np.sqrt(1 - ((EV1p/EV1)**(2/3))) ) + h  # (Eq. 21)
H_S3    = np.full(len(D_S3), h)  # lights with same hights as Lan3
H_S5    = np.full(len(D_S5), h)  # lights with same hights as Lan3
H_simul3 = d_simul3 * (EVS3/abs(EVS35)) + h  # (Eq. 15)
H_simul5 = d_simul5 * (EVS5/abs(EVS35)) + h  # (Eq. 15)



# Line between light and lancube (Orthogonal) 
EO1  = df1['lux'].iloc[idx_E1].values
EO3 = abs(df3['lux'].iloc[idx_E3].values)
EO5 = abs(df5['lux'].iloc[idx_E5].values)
EOS3 = df1['lux'].iloc[idx_ES1].values * (np.sqrt((H_simul3 - h)**2 + H_simul3**2))\
        / (H_simul3 - h)  # (Eq. 16)
EOS5 = df1['lux'].iloc[idx_ES1].values * (np.sqrt((H_simul5 - h)**2 + H_simul5**2))\
        /(H_simul5 - h)  # (Eq. 16)

# Concatenate scenarios data
d = np.concatenate([d_simul, d_S1, d_S35])
D = np.concatenate([D_simul, D_S1, D_S35])
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



# GAUSSIAN FILTER AND RECALCULATIONS OF THE DATAS
df_update = gaussian_filter(df_invent, h)

# Use mask only for none h height
mask_H = df_update['H'].values > h
H_p = df_update['H'].values[mask_H]
flux = flux[mask_H]
H = H[mask_H]
d = d[mask_H]
lat_peak =lat_peak[mask_H]
lon_peak = lon_peak[mask_H]
lat_lights = lat_lights[mask_H]
lon_lights = lon_lights[mask_H]


# ReDo calculations with update heights
flux_p = flux*( (H_p-h) / (H-h) )**2
d_p = d*(H_p-h) / (H-h)
lat_lights_p = lat_peak + (lat_lights-lat_peak) * (H_p-h)/(H-h)
lon_lights_p = lon_peak + (lon_lights-lon_peak) * (H_p-h)/(H-h)


# Update inventory
df_invent.loc[mask_H, 'lat_lights'] = lat_lights_p
df_invent.loc[mask_H, 'lon_lights'] = lon_lights_p
df_invent.loc[mask_H, 'd'] = d_p
df_invent.loc[mask_H, 'flux'] = flux_p

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




# -----------------------------------------------------------------------------------

filename = 'lan3_invent_2022-02-07-08.csv'
df_invent = pd.read_csv(f'inventaires/{filename}')

def distance(lat_obs, lon_obs, arr_lats, arr_lon):
    return np.sqrt((lat_obs - arr_lats)**2+((lon_obs-arr_lon)*np.cos(arr_lon))**2)


def find_close_lights(df, df_invent, nb=10, h=2):    
    
    lat, lon = [], []
    
    for i in df.index.values:
        # i = df.index.values[0]
        
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
        
        # sort_dist = list_dist.copy()
        # sort_dist.sort()
        # sort_dist = np.array(sort_dist)
        # sort_dist = sort_dist[np.logical_not(np.isnan(sort_dist))] # remove nan from array
        # idx_closest = [list_dist.index(k) for k in sort_dist[1:nb+1]] # get closest idx

        lat.append(df_invent['lat_lights'].iloc[idx_dist].values)
        lon.append(df_invent['lon_lights'].iloc[idx_dist].values)
        
        # Mean of the 10 closest lights
        df = df_invent.iloc[idx_dist]
        update_H = df['H'].mean()

        # H = df['H'].values
        # idx_mask = np.argwhere(H > h).flatten()
        # y, x, _ = plt.hist(H, bins=int(H.max()//2), alpha=.3)
        # x=(x[1:]+x[:-1])/2  # for len(x)==len(y) 
        # update_H.append( x[np.argmax(y)] )
    
    return update_H, lat, lon
    

update_H = []
df_H = df_invent[ (df_invent['flux'] > 30000) & (df_invent['H'] > 2) ]

update_H, lat, lon = find_close_lights(df_H, df_invent, nb=10)

lat = np.concatenate( lat, axis=0 )
lon = np.concatenate( lon, axis=0 )
df_latlon = pd.DataFrame({'lat':lat, 'lon':lon})
df_latlon.to_csv('latlon_close.csv')

df_invent['H'].iloc[df_H.index.values] = update_H



# ----------------------------------------------------------------------------------

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
