#!/usr/bin/env python3

# Filter the data measured by the lancube
# Author : Julien-Pierre Houle
# Last update : August 2021

import os
import numpy as np
import pandas as pd
import osmnx as ox
import progressbar
from datetime import datetime
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage.interpolation import shift


# PARAMETERS
PATH_DATAS = "Lan3_mtl"
light_folder = 'Lights/' # path to the lights folder
street_distance = 5 # Maximal distance(m) between coord and street
h = 2   # height of the lancube in meters
K = 1  # K factor



# Load Lights tech clouds
print('Loading lights tech cloud')
list_tech = []
frames = []

for filename in os.listdir(light_folder):
    df = pd.read_csv(f"{light_folder}/{filename}")
    list_tech += [filename[:-4]] * len(df)
    frames.append(df)

df_lights = pd.concat(frames, ignore_index=True)
lights_cloud = np.array((df_lights['RC_avg'].values,
                         df_lights['GC_avg'].values,
                         df_lights['BC_avg'].values))


df = pd.read_csv(f"{PATH_DATAS}/2021-06-07.csv", sep=',', error_bad_lines=False)


def delete_sequence(index_to_delete):
    """ Function to return the full sequence of sensors to delete when a value is filtered. """

    print('Filtering data...')
    sequence_to_delete = []
    matching_sensor = df["Sensor"].iloc[index_to_delete].values.tolist()
    sensor_number = [int(s[-1]) for s in matching_sensor] # nb associate to the sensor (between 1-5)

    bar = progressbar.ProgressBar(maxval=len(index_to_delete)).start()
    for idx, index_error in enumerate(index_to_delete):
        sensor = sensor_number[idx] # sensor number
        diff_sup = 5-sensor

        if index_error not in sequence_to_delete:
            sequence_sup = list(range(index_error, index_error+diff_sup+1))
            sequence_inf = list(reversed((range(index_error-sensor+1, index_error))))

            sequence_to_delete += (sequence_sup+sequence_inf)
            bar.update(idx)

    return sequence_to_delete


# -------------------- Preprocess data --------------------

df['Time'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute','Second']],\
                            format="Y-%m-%d %H-%M-%S")
df=df.drop(columns=['Year','Month','Day','Hour','Minute', 'Second'])


df['Value'] = df['Clear'] / df['Gain'] / df['Acquisition time (ms)']
df['MA'] = df["Value"].rolling(window=3).mean()


# -------------------- Data Filter --------------------

index_to_delete = []

# 1. Filter GPS disconnected and abberant lat, lon
mask_zeros = (df['Latitude'] == 0.0) & (df['Longitude'] == 0.0)
index_to_delete += df[mask_zeros].index.tolist()  # Delete 0.0 value when GPS is disconnect

avg_lat, avg_lon = df['Latitude'].mean(), df['Longitude'].mean()
mask_lat = (df['Latitude']  > avg_lat+2) & (df['Latitude']  < avg_lat-2)
mask_lon = (df['Longitude'] > avg_lon+2) & (df['Longitude'] < avg_lon-2)
index_to_delete += df[mask_lat & mask_lon].index.tolist()



# 2. Filter distance with previous point
# The maximum distance between 2 points accepted is 5m. Distance between 2 coords:
# dy = y'-y ≈ (π/180)* R∆φ and dx = x'-x ≈ (π/180) * R∆λ*sin(φ).
dy = np.pi/180 * 6373000 * df['Latitude'].diff()
dx = np.pi/180 * 6373000 * df['Longitude'].diff() * np.sin(df['Latitude'])
D = np.sqrt(dx**2 + dy**2)
index_to_delete += D[D > 5].index.tolist()



# 3. Filter Over Expose S1 and abberant values
index_to_delete += df[(df['Sensor'] == "S1") & (df['Flag'] == 'OE')].index.tolist()
index_to_delete += df[(df['Sensor'] == "S3") & (df['Flag'] == 'OE')].index.tolist()
index_to_delete += df[(df['Sensor'] == "S5") & (df['Flag'] == 'OE')].index.tolist()
index_to_delete = list(set(index_to_delete)) # make sure to have unique index value

drop_value_error = delete_sequence(index_to_delete)
df = df.drop(drop_value_error).reset_index(drop=True)



# # 4. Filter distance with road
# # Remove points if the distance with the road greater than 5m du to GPS interference
# def get_bearing(lat1, lon1, lat2, lon2):
#     x = np.cos(np.deg2rad(lat2)) * np.sin(np.deg2rad(lon2-lon1))
#     y = np.cos(np.deg2rad(lat1)) * np.sin(np.deg2rad(lat2)) - \
#         np.sin(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.cos(np.deg2rad(lon2-lon1))
#     return np.rad2deg(np.arctan2(x,y))


# print("    Loading graph")
# lats = df['Latitude']
# lons = df['Longitude']
# Graph = ox.graph_from_bbox(
#     north=max(lats)+0.0001,
#     south=min(lats)-0.0001,
#     east=max(lons)+0.0001,
#     west=min(lons)-0.0001,
#     network_type='drive',
#     simplify=False,
#     retain_all=True,
#     truncate_by_edge=True,
#     clean_periphery=True)
#
# Graph = ox.utils_graph.get_undirected(Graph)
# Graph = ox.bearing.add_edge_bearings(Graph, precision=0)
# Graph = ox.projection.project_graph(Graph, to_crs='epsg:2949')
# nodes, edges = ox.graph_to_gdfs(Graph)
# df_routes = edges.filter(['name','bearing','geometry'], axis=1)
#
# inProj =  Proj('epsg:4326')
# outProj = Proj('epsg:2949')
# X, Y = transform(inProj, outProj, lons, lats, always_xy=True)
#
# print("    Get distance with nearest edges")
# edges_ID, dist = ox.distance.nearest_edges(Graph, X, Y, interpolate=True, return_dist=True)
# df["Distance_street"] = dist
# index_to_delete += df[df["Distance_street"] > street_distance].index.values.tolist()



# -------------------- Data Manipulation--------------------

# Calculate traveled distance
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2949", always_xy=True)
X, Y = transformer.transform(df['Longitude'], df['Latitude'])

dX, dY = np.diff(X), np.diff(Y)
dDist = np.sqrt((dX**2) + (dY**2)) # pytagore
traveled_distance = np.cumsum(dDist) # cumulative sum of the distance
df["Traveled Distance"] = np.insert(traveled_distance, 0, np.nan).tolist() # add nan

df1 = df[df['Sensor'] == 'S1']
df3 = df[df['Sensor'] == 'S3']
df5 = df[df['Sensor'] == 'S5']
df1 = df1.drop(0)

dfS = pd.DataFrame({'Value':    (df3['Value'].values - df5['Value'].values),
                    'Distance':   df3['Traveled Distance'].values,
                    'BC_3': (df3['Blue']  / df3['Clear']).values,
                    'RC_3': (df3['Red']   / df3['Clear']).values,
                    'GC_3': (df3['Green'] / df3['Clear']).values,
                    'BC_5': (df5['Blue']  / df5['Clear']).values,
                    'RC_5': (df5['Red']   / df5['Clear']).values,
                    'GC_5': (df5['Green'] / df5['Clear']).values})

S1_value = df1['Value'].values
S1_dist  = df1['Traveled Distance'].values

# Moving Average
df1['MA'] = df1["Value"].rolling(window=3, center=True).mean()
dfS['MA'] = dfS['Value'].rolling(window=3, center=True).mean()




# ------------ Find index peaks S1 & S3-S5 (side) ------------

print('Finding peaks and primes..')
idx_EV1, _m = find_peaks(df1['MA'], height=0.04, prominence=0.02)
idx_EVS, _m = find_peaks(abs(dfS['MA']), height=0.04, prominence=0.02)

# S1 & S3-S5 peaks & distance values
EV1, ED1 = S1_value[idx_EV1], S1_dist[idx_EV1]
EVS, EDS = dfS['Value'].iloc[idx_EVS], dfS['Distance'].iloc[idx_EVS].values


idx_simul   = []
idx_simul_side = []
idx_S1  = []

# Check for (S3-S5) peak close to S1 peak
for pos in idx_EV1:
    d = S1_dist[pos] # distance peak S1

    simul_dist = EDS[((d-3) < EDS) & (EDS < (d+3))] # EDS in a 5m interval from S1 peak

    if len(simul_dist) == 1:  # high light fixture
        idx_simul.append(pos)
        idx_simul_side.append(dfS['Distance'].tolist().index(simul_dist[0]))

    if len(simul_dist) == 0: # peak only in S1
        idx_S1.append(pos)


# Get index low lights (peak only in S3-S5)
idx_S35 = [elem for elem in idx_EVS if elem not in idx_simul_side]



def find_prime(index_peaks, values, dist):
    """ Find prime value +- 2 points close to the index peak.
        Will take the left value if available else the right one. """

    idx = np.array(index_peaks)
    mask_L = (values[idx-2] < values[idx]) & (dist[idx]-dist[idx-2] < 15)
    mask_R = (values[idx+2] < values[idx]) & (dist[idx+2]-dist[idx] < 15)
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
prime_S35,   peaks_S35,   _del     = find_prime(idx_S35, dfS['Value'].values, dfS['Distance'].values)


# Get peaks and prime values
EV, ED = S1_value[peaks_simul], S1_dist[peaks_simul]
EVp, EDp = S1_value[prime_simul], S1_dist[prime_simul]
EV_side  = dfS['Value'].iloc[np.array(idx_simul_side)[deleting]].values

EV1, ED1 = S1_value[peaks_S1], S1_dist[peaks_S1]
EV1p, ED1p = S1_value[prime_S1], S1_dist[prime_S1]

EV35, ED35 = dfS['Value'].iloc[peaks_S35].values, dfS['Distance'].iloc[peaks_S35].values
EV35p, ED35p = dfS['Value'].iloc[prime_S35].values, dfS['Distance'].iloc[prime_S35].values




# ------------ Getting lights characteristics ------------
print('Getting lights characteristics..')

# Distance between peaks and prime
D_simul = ED - EDp
D_S1    = ED1 - ED1p
D_S35   = ED35 - ED35p

# Horizontal distance between Lan3 and lights fixture at t.
d_simul = D_simul / ( (EV/EVp)**(2/3) * ( ((EV**2)/ (EV_side)**2)+1 ) - ((EV**2)/ (EV_side)**2)-1)
d_S1    = np.full(len(D_S1), 0)
d_S35   = D_S35 * (EV35p/EV35)**(1/3) / np.sqrt(1 - (EV35p/EV35)**(2/3))

# Light fixture height.
H_simul = d_simul * (EV/abs(EV_side))-h
H_S1    =  (D_S1 * ((EV1p/EV1)**(1/3)) / np.sqrt(1 - ((EV1p/EV1)**(2/3))) ) + h
H_S35   = np.full(len(D_S35), h)

# Concatenate scenarios data
D = np.concatenate([D_simul, D_S1, D_S35])
d = np.concatenate([d_simul, d_S1, d_S35])
H = np.concatenate([H_simul, H_S1, H_S35])
E_val = np.concatenate([EV, EV1, EV35])

# Equation 16
E_perpendicular = E_val * (np.sqrt((H - h)**2 + d**2)) / (H - h)



# Find nearest point in the lights technologie cloud
peak_simul_S1 = np.concatenate([peaks_simul, peaks_S1])
M_simul_S1 = np.stack( ((df['Blue']  / df['Clear']).iloc[peak_simul_S1].values,
                        (df['Green'] / df['Clear']).iloc[peak_simul_S1].values,
                        (df['Red']   / df['Clear']).iloc[peak_simul_S1].values), axis=-1)


S35_peaks_val = dfS['Value'].iloc[peaks_S35]
idx_S3 = S35_peaks_val[S35_peaks_val > 0].index.values
idx_S5 = S35_peaks_val[S35_peaks_val < 0].index.values

# Convatening both dataframe (S3 & S5) while maintening index order
M_S35 = np.stack((
        pd.concat([dfS['BC_3'].iloc[idx_S3], dfS['BC_5'].iloc[idx_S5]], sort=False).sort_index().values,
        pd.concat([dfS['GC_3'].iloc[idx_S3], dfS['GC_5'].iloc[idx_S5]], sort=False).sort_index().values,
        pd.concat([dfS['RC_3'].iloc[idx_S3], dfS['RC_5'].iloc[idx_S5]], sort=False).sort_index().values),
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

flux = (2*np.pi*K*E_perpendicular) * (d**2 + (H-h)**2) / (1 - U)



# ----------------- Writing data -----------------
print('Writing data...')

lats = df1['Latitude'].iloc[peak_simul_S1].tolist() + \
       df3['Latitude'].iloc[peaks_S35].tolist()

lons = df1['Longitude'].iloc[peak_simul_S1].tolist() + \
       df3['Longitude'].iloc[peaks_S35].tolist()

df_invent = pd.DataFrame({
            'lat'   : lats,
            'lon'   : lons,
            'D'     : list(np.concatenate([D_simul, D_S1, D_S35])),
            'd'     : list(np.concatenate([d_simul, d_S1, d_S35])),
            'H'     : list(np.concatenate([H_simul, H_S1, H_S35])),
            'tech'  : list(lights_tech),
            'E_perp': list(E_perpendicular),
            'flux'  : list(flux)} )

df_invent['h'] = h
# df.to_csv('lan3_inventory.csv', index=False, sep=' ')
print('\nDone.')



# ********************************************************************

plt.plot(df1['Traveled Distance'], df1['Value'], label='S1')
plt.plot(dfS['Distance'], dfS['Values'], label='S3-S5')

# MA
plt.plot(df1['Traveled Distance'], df1['MA'], label='S1 MA')
plt.plot(dfS['Distance'], dfS['MA'], label='Side MA')

# PEAKS
plt.plot(ED1, EV1, 'o', c='red')
plt.plot(EDp, EVp, 'o', c='pink') # Prime

plt.legend()
plt.show()
