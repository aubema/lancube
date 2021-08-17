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


# PARAMETERS
PATH_DATAS = "Lan3_mtl"
light_folder = 'Lights/' # path to the lights folder
street_distance = 5 # Maximal distance(m) between coord and street
h = 2   # height of the lancube in meters



# Load Lights tech clouds
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

dfS = pd.DataFrame({'Values':    (df3['Value'].values - df5['Value'].values),
                    'Distance':   df3['Traveled Distance'].values,
                    'Init_index': df3.index.values})

# Moving Average
df1['MA'] = df1["Value"].rolling(window=3, center=True).mean()
dfS['MA'] = dfS['Values'].rolling(window=3, center=True).mean()


S1_value = df1['Value'].values
S1_dist  = df1['Traveled Distance'].values


# Find index peaks S1 & S3-S5 (side)
idx_EV1, _m = find_peaks(df1['MA'], height=0.04, prominence=0.02)
idx_EVS, _m = find_peaks(abs(dfS['MA']), height=0.04, prominence=0.02)

# S1 & S3-S5 peaks & distance values
EV1, ED1 = S1_value[idx_EV1], S1_dist[idx_EV1]
EVS, EDS = dfS['Values'].iloc[idx_EVS], dfS['Distance'].iloc[idx_EVS].values


idx_simul_S1   = []
idx_simul_side = []
index_S1_only  = []

# Check for (S3-S5) peak close to S1 peak
for pos in idx_EV1:
    d = S1_dist[pos] # distance peak S1

    simul_dist = EDS[((d-3) < EDS) & (EDS < (d+3))] # EDS in a 5m interval from S1 peak

    if len(simul_dist) == 1:  # high light fixture
        idx_simul_S1.append(pos)
        idx_simul_side.append(dfS['Distance'].tolist().index(simul_dist[0]))

    if len(simul_dist) == 0: # peak only in S1
        index_S1_only.append(pos)


# Get index low lights (peak only in S3-S5)
idx_low_lights = [elem for elem in idx_EVS if elem not in idx_simul_side]


idx_left = []
idx_right = []

# Find fraction values close to each peak
for index, pos in enumerate(idx_simul_S1):

    if S1_value[pos-2] < S1_value[pos] and (S1_dist[pos]-S1_dist[pos-2]) < 15:
        idx_left.append(pos-2)
    else:
        idx_left.append(np.nan)

    if S1_value[pos+2] < S1_value[pos] and (S1_dist[pos+2]-S1_dist[pos]) < 15:
        idx_right.append(pos+2)
    else:
        idx_right.append(np.nan)


# Take left value if availaible else take right
nan_min = np.nanmin([idx_left, idx_right], axis=0)
mask = ~np.isnan(nan_min)
idx_prime = nan_min[mask].astype(int)
idx_peaks = np.array(idx_simul_S1)[mask].astype(int)


# Get simultanous peaks values
EV1, ED1 = S1_value[idx_peaks], S1_dist[idx_peaks]
EVp, EDp = S1_value[idx_prime], S1_dist[idx_prime]
EVS, EDS = dfS['Values'].iloc[idx_peaks].values, dfS['Distance'].iloc[idx_peaks].values


# # Equation 14: Horizontal distance between Lan3 and lights fixture at t.
D = ED1 - EDp # distance between EV1 and EV1p
d = D / ( (EV1/EVp)**(2/3) * ( ((EV1**2)/ (EVS)**2)+1 ) - ((EV1**2)/ (EVS)**2)-1)


# Equation 15: Light fixture height.
H = d*(EV1/abs(EVS))-h


# Equation 16: Illuminance normal line
dist_light = np.sqrt((H - h)**2 + d**2) # distance between lan3 and light
E_illum = (EV1 * dist_light) / (H - h)


# RC, GC, BC = np.zeros(len(dfS)), np.zeros(len(dfS)), np.zeros(len(dfS))
# index_S3 = dfS[dfS['Values'] > 0].index.values
# index_S5 = dfS['Init_index'][dfS['Values'] < 0].values
# df3['Blue'].iloc[index_S3]

# Find nearest point Lights technologie cloud
BC = (df1['Blue']  / df1['Clear']).iloc[idx_peaks].values
GC = (df1['Green'] / df1['Clear']).iloc[idx_peaks].values
RC = (df1['Red']   / df1['Clear']).iloc[idx_peaks].values
M = np.stack((BC, GC, RC), axis=-1)

distance = np.sum((M[:,:,None] - lights_cloud) **2, 1)
idx_closest = np.argmin(distance, 1)
light_tech  = np.array(list_tech)[idx_closest] # return closest light technologie



# Writing data
data_peaks = {'lat'   : df1['Latitude'].iloc[idx_peaks].tolist(),
              'lon'   : df1['Longitude'].iloc[idx_peaks].tolist(),
              'D'     : list(D),
              'd'     : list(d),
              'H'     : list(H),
              'technologie': list(light_tech)}

data_S1 = {'lat'  : df1['Latitude'].iloc[index_S1_only].tolist(),
          'lon'   : df1['Longitude'].iloc[index_S1_only].tolist()}

data_side = {'lat'  : df1['Latitude'].iloc[idx_low_lights].tolist(),
             'lon'  : df1['Longitude'].iloc[idx_low_lights].tolist()}

df_peaks = pd.DataFrame.from_dict(data_peaks)
df_S1    = pd.DataFrame.from_dict(data_S1)
df_side  = pd.DataFrame.from_dict(data_side)


df_invent = pd.concat([df_peaks, df_S1, df_side]).reset_index(drop=True)
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


# plt.plot(df1['Traveled Distance'].iloc[idx_EV1], df1['Value'].iloc[idx_EV1], 'o')
# plt.plot(dfS['Distance'].iloc[idx_EVS], dfS['Values'].iloc[idx_EVS], 'o')


# Matched
# plt.plot(ED1, EV1, 'x', c='r')
# plt.plot(EDS, EVS, 'x', c='r')

# Not Matched
# plt.plot(S1_dist[index_S1_only], S1_value[index_S1_only], 'x', c='b', label='S1 only')
# plt.plot(side_d[idx_low_lights], side_v[idx_low_lights], 'x', c='g',  label='Side only')

plt.legend()
plt.show()
