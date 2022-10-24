#!/usr/bin/env python3

# Cleaning the data measured by the lancube
# Author : Julien-Pierre Houle


import numpy as np
import pandas as pd
import osmnx as ox
from pyproj import CRS, Transformer

def delete_sequence(df, index_to_delete):
    """ Function to return the full sequence of sensors to delete when a value is filtered. """

    print('Filtering data.')
    seq_to_delete = []
    sensor = df["Sensor"].iloc[index_to_delete].values.tolist()    
   
    for idx, index_error in enumerate(index_to_delete):
                
        if sensor[idx] == 'S1':
            seq_to_delete += list(range(index_error, index_error+3))
            
        if sensor[idx] == 'S3':
            seq_to_delete += [index_error-1, index_error, index_error+1]
            
        if sensor[idx] == 'S5':
            seq_to_delete += list(range(index_error, index_error-3, -1))
        
    seq_to_delete = list(set(seq_to_delete))

    return seq_to_delete




def cleaning_data(df):
    """ Cleaning the data measured by the lancube. """
    
    # PATH_DATAS = "Data"
    # df = pd.read_csv(f"{PATH_DATAS}/Corr_2021-10-28-temp.csv", sep=',', on_bad_lines='skip')
    # df['Value'] = df['lux']
    
    
    df['Time'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute','Second']],
                                                         format="Y-%m-%d %H-%M-%S")
    df = df.drop(columns=['Year','Month','Day','Hour','Minute', 'Second'])
    
    df = df.drop( df[ (df.Sensor=='S2') | (df.Sensor == 'S4')].index ) # delete S2 & S4 sensor
    df = df.dropna(how='all')  # delete row if all values are NaN
    df = df.reset_index(drop=True) # reset a normal index


    # Verify sensor order
    index = df.index.values
    sensors_nb = np.array([int(s[-1]) for s in df['Sensor'].tolist()]) # value between 1-5
    diff_prev = np.diff(sensors_nb, prepend=0) # diff with previous values
    idx_error = index[(diff_prev != 2) & (diff_prev != -4)] # error if diff not 2 or -4
    idx_error = np.append(idx_error, df.index[-1]) # check last sensors
    

    # Cleaning the sensor order
    index_to_delete = []
    for i in range(len(idx_error)):
        
        idx = idx_error[i]
        index_to_delete.append(idx)
        

        # Low sensors
        prev_sensor = 0
        while (prev_sensor != 'S5') and (idx > 0):
            idx -= 1
            prev_sensor = df['Sensor'].iloc[idx]
            
            if prev_sensor != 'S5':
                index_to_delete.append(idx)

        idx = idx_error[i]

        # Top sensors
        next_sensor = 0
        while (next_sensor != 'S1') and (idx < len(df)-1):
            idx += 1
            next_sensor = df['Sensor'].iloc[idx]
                        
            if next_sensor != 'S1':
                index_to_delete.append(idx)

    index_to_delete.sort()
    df = df.drop(index_to_delete).reset_index(drop=True)



    # ----------------- Filtering data ----------------

    index_to_delete = []

    # 1. Filter GPS disconnected and abberant lat, lon
    mask_zeros = (df['Latitude'] == 0.0) & (df['Longitude'] == 0.0)
    index_to_delete += df[mask_zeros].index.tolist()  # Delete 0.0 value when GPS is disconnect

    avg_lat, avg_lon = df['Latitude'].mean(), df['Longitude'].mean()
    mask_lat = (df['Latitude']  > avg_lat+2) & (df['Latitude']  < avg_lat-2)
    mask_lon = (df['Longitude'] > avg_lon+2) & (df['Longitude'] < avg_lon-2)
    index_to_delete += df[mask_lat & mask_lon].index.tolist()


    # index_to_delete += df['Value'][df['Value'].abs() > 10].index.tolist()



    # 2. Filter distance with previous point
    # The maximum distance between 2 points accepted is 5m. Distance between 2 coords:
    # dy = y'-y ≈ (π/180)* R∆φ and dx = x'-x ≈ (π/180) * R∆λ*sin(φ).
    dy = np.pi/180 * 6373000 * df['Latitude'].diff()
    dx = np.pi/180 * 6373000 * df['Longitude'].diff() * np.sin(df['Latitude'])
    D = np.sqrt(dx**2 + dy**2)
    
    # 20m correspond 120 km/h (condition tres noire)
    index_to_delete += D[D > 20].index.tolist()
    
    
    # Check for negative lux to delete
    mask_lux = (df['lux'] < 0)
    index_to_delete += df[mask_lux].index.tolist()


    # 3. Filter Over Expose S1 and abberant values
    index_to_delete += df[(df['Sensor'] == "S1") & (df['Flag'] == 'OE')].index.tolist()
    index_to_delete += df[(df['Sensor'] == "S3") & (df['Flag'] == 'OE')].index.tolist()
    index_to_delete += df[(df['Sensor'] == "S5") & (df['Flag'] == 'OE')].index.tolist()
    index_to_delete = list(set(index_to_delete)) # make sure to have unique index value


    drop_value_error = delete_sequence(df, index_to_delete)
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
    dDist = np.sqrt((dX**2) + (dY**2)) # pythagore
    traveled_distance = np.cumsum(dDist) # cumulative sum of the distance
    df["distance"] = np.insert(traveled_distance, 0, np.nan).tolist() # add nan first elem

    df1 = df[df['Sensor'] == 'S1']
    df3 = df[df['Sensor'] == 'S3']
    df5 = df[df['Sensor'] == 'S5']
    
    # Rename columns
    df1.rename(columns = {'Latitude':'lat', 'Longitude':'lon'}, inplace=True)
    df3.rename(columns = {'Latitude':'lat', 'Longitude':'lon'}, inplace=True)
    df5.rename(columns = {'Latitude':'lat', 'Longitude':'lon'}, inplace=True)
    
    # Reset index
    df1 = df1.reset_index()
    df3 = df3.reset_index()
    df5 = df5.reset_index()


    return df1, df3, df5



# PATH_DATAS = "Data"
# df = pd.read_csv(f"{PATH_DATAS}/Corr_2021-10-28-temp.csv", sep=',', on_bad_lines='skip')
# df['Value'] = df['lux']


# df = pd.read_csv('Data/Mtl_june_2021/2021-06-07.csv')
# df.rename(columns={'Acquisition time (ms)': 'AcquisitionTime'}, inplace=True)
# df['Value'] = df['Clear'] / df['Gain'] / df['AcquisitionTime']

# PATH_DATAS = "Data/Mtl_2"
# df = pd.read_csv(f"{PATH_DATAS}/Corr_2021-06-07.csv", sep=',', on_bad_lines='skip')
# df['Value'] = df['Clear'] / df['Gain'] / df['AcquisitionTime']


# df1, df3, df5 = cleaning_data(df)


# import matplotlib.pyplot as plt
# plt.plot(df1['Traveled Distance'], df1['Value'])
# plt.show()