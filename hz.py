import csv
import pandas as pd
import time

while True:
    data = pd.read_csv("/var/www/html/data/gps.txt")
    data_list = data.values.tolist()

    t = data_list[-1][0]
    tmoins = data_list[-2][0]
    lat_t = data_list[-1][1]
    lat_t_moins = data_list[-2][1]
    lon_t = data_list[-1][2]
    lon_t_moins = data_list[-2][2]


    futur_lat = (lat_t-lat_t_moins)/(t-tmoins)
    futur_lon = (lon_t-lon_t_moins)/(t-tmoins)

    print(futur_lat)
    print(futur_lon)
    time.sleep(0.2)