# ******************************************************************************
#                   Filter the data measured by the lancube
#
# Author : Julien-Pierre Houle
# Date   : June 2021
# ******************************************************************************


import numpy as np
import pandas as pd
import osmnx as ox
import progressbar
from datetime import datetime
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


# PARAMETERS
PATH_DATAS = "Lan3_mtl"
street_distance = 5 # Maximal distance(m) between coord and street
lan3_height = 2     # height of the lancube in meters
EVp_fract  = 0.8   # Fraction of the peak value diminuated
EVp_range  = 20    # range of position before and after to find EV_p

df = pd.read_csv(f"{PATH_DATAS}/2021-06-07.csv", sep=',', error_bad_lines=False)


def delete_sequence(index_to_delete):
    """ Function to return the full sequence of sensors to delete when a value is filtered. """

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
df['MA'] = df["Value"].rolling(window=5).mean()


# -------------------- Data Filter --------------------

index_to_delete = []

# 1. Filter GPS disconnected and abberant lat, lon
mask_zeros = (df['Latitude'] == 0.0) & (df['Longitude'] == 0.0)
index_to_delete += df[mask_zeros].index.values.tolist()  # Delete 0.0 value when GPS is disconnect

avg_lat, avg_lon = df['Latitude'].mean(), df['Longitude'].mean()
mask_lat = (df['Latitude']  > avg_lat+2) & (df['Latitude']  < avg_lat-2)
mask_lon = (df['Longitude'] > avg_lon+2) & (df['Longitude'] < avg_lon-2)
index_to_delete += df[mask_lat & mask_lon].index.values.tolist()


# 2. Filter distance with previous point
# The maximum distance between 2 points accepted is 5m. The distance between 2 coords is
# calculate with dy = y'-y ≈ (π/180)* R∆φ and dx = x'-x ≈ (π/180) * R∆λ*sin(φ).
dy = np.pi/180 * 6373000 * df['Latitude'].diff()
dx = np.pi/180 * 6373000 * df['Longitude'].diff() * np.sin(df['Latitude'])
D = np.sqrt(dx**2 + dy**2)
index_to_delete += D[D > 5].index.values.tolist()




# 3. Filter Over Expose S1 and abberant values
index_to_delete += df[(df['Sensor'] == "S1") & (df['Flag'] == 'OE')].index.values.tolist()
index_to_delete += df[(df['Sensor'] == "S3") & (df['Flag'] == 'OE')].index.values.tolist()
index_to_delete += df[(df['Sensor'] == "S5") & (df['Flag'] == 'OE')].index.values.tolist()
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

#
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


# Find index Over Expose S3 et S5
df1 = df[df['Sensor'] == "S1"].reset_index(drop=True)
df3 = df[df['Sensor'] == "S3"].reset_index(drop=True)
df5 = df[df['Sensor'] == "S5"].reset_index(drop=True)

idx_EV1, _m = find_peaks(df1['MA'], distance=10, prominence=0.03)
idx_EV3, _m = find_peaks(df3['MA'], distance=10, prominence=0.03)


EV1 = df1["Value"].iloc[idx_peaks_S1]
EV3 = df3["Value"].iloc[idx_peaks_S3]
peaks_dist_S1 = df1["Traveled Distance"].iloc[idx_EV1]
peaks_dist_S3 = df3["Traveled Distance"].iloc[idx_EV3]

distance = df1["Traveled Distance"].values.tolist()
values = [df1["Value"].values.tolist(),  df3["Value"].values.tolist()]


# Find distance between peaks and 80% of the peaks value in a 40 positions range
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# ** Pas besoin de EV3P??
EVP = []
for index, EV in enumerate([EV1, EV3]):
    peak_P = np.zeros(len(EV))
    for idx, pos in enumerate(EV.index.tolist()):
        arr = values[index][pos-EVp_range:pos+EVp_range]
        val = EV.iloc[idx] * EVp_fract # fraction of peak value

        peak_P[idx] = find_nearest(arr, val)
    EVP.append(peak_P)
EV1_p, EV3_p = EVP[0], EVP[1]



# Find horizontal distance between Lan3 and lights fixture
# Equation: **
D = peaks_dist_S1.values - np.array((close_peak_S1))
d = D / ( (EV1/EV1_p)**(2/3) * ((EV1**2)/(EV3**2)+1) - ((EV1**2)/(EV3**2)-1) )


# Equation 15*: Calculate lamps height
H = d*(EV1/EV3)-lan3_height




# Write results to file
data = {'lat' : df1['Latitude'].iloc[idx_EV1].values.tolist(),
        'lon' : df1['Longitude'].iloc[idx_EV1].values.tolist(),
        'd'   :
}

df_results = pd.DataFrame(files['coord'], columns = ['#lat', 'lon', 'd', 'H'])
df['pow'] = POW.tolist()
df['hobs'] = hobs.tolist()
df['dobs'] = dobs.tolist()
df['fobs'] = fobs.tolist()z
df['hlamp'] = hlamp.tolist()
df['spct'] = arr_spcts.tolist()
df['lop'] = arr_lops.tolist()
df.to_csv(inventory_name, index=False, sep=' ')
print('Done')



# # Find nearest matched peak for the other array
# if len(index_peaks_S1) < len(index_peaks_S3):
#     index_peaks_S3 = np.array([find_nearest(index_peaks_S3, v) for v in index_peaks_S1])
# else:
#     index_peaks_S1 = np.array([find_nearest(index_peaks_S1, v) for v in index_peaks_S3])
#
# values_peaks_S1 = df.iloc[index_peaks_S1]['MA'].values
# values_peaks_S3 =  df.iloc[index_peaks_S3]['MA'].values
#
# # Determine highest values for each pair of peaks
# index_lights_higher = index_peaks_S1[values_peaks_S1 > values_peaks_S3]
# index_lights_lower  = index_peaks_S1[values_peaks_S1 < values_peaks_S3]

# ********************************************************************

# Plot S3 - S5
diff = df3['Value'].to_numpy() - df5['Value'].to_numpy()
plt.plot(x, np.delete(df1['Value'].values, 0), label='S1')
plt.plot(x,  diff, label='S3 - S5')
plt.legend()
plt.ion()
plt.show()


# Plot higher and lower lights
plt.plot(df1['Traveled Distance'], df1['MA'], label='S1')
plt.plot(df3['Traveled Distance'], df3['MA'], label='S3')
plt.plot(df['Traveled Distance'].iloc[index_lights_higher], df['MA'].iloc[index_lights_higher], "x")
plt.plot(df['Traveled Distance'].iloc[index_lights_lower],  df['MA'].iloc[index_lights_lower], "x")

plt.legend()
plt.ion()
plt.show()
