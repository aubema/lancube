import csv
import pandas as pd
import time
from datetime import datetime

while True:
    today = str(datetime.now())
    hour = int(today[11:13])
    minute = int(today[14:16])
    second = float(today[17:])
    times = hour*3600 + minute*60 + second

    data = pd.read_csv("/var/www/html/data/gps.txt")
    data_list = data.values.tolist()

    t = data_list[-1][0]
    tmoins = data_list[-2][0]
    lat_t = data_list[-1][1]
    lat_t_moins = data_list[-2][1]
    lon_t = data_list[-1][2]
    lon_t_moins = data_list[-2][2]


    futur_lat = (lat_t-lat_t_moins)/(t-tmoins)*(times-t)+lat_t
    futur_lon = (lon_t-lon_t_moins)/(t-tmoins)*(times-t)+lon_t

    print(futur_lat)
    print(futur_lon)
    time.sleep(0.2)