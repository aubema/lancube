import subprocess
import numpy as np
import csv
from datetime import datetime
import os

data = open("/var/www/html/data/gps.csv", 'w', newline='')

today = str(datetime.now())
year = today[0:4]
month = today[5:7]
day = today[8:10]

writer_data = csv.writer(data)
open("/var/www/html/data/live/gps_log.txt", 'w').close()

while True:
    today = str(datetime.now())
    hour = int(today[11:13])
    minute = int(today[14:16])
    second = float(today[17:])
    time = hour*3600 + minute*60 + second

    log = open("/var/www/html/data/live/gps_log.txt", 'a')
    data = open("/var/www/html/data/gps.txt", 'a', newline='')
    
    raw = subprocess.Popen("/usr/bin/gpspipe -w -n 5", shell=True, stdout=subprocess.PIPE)
    raw_out = raw.stdout.read()

    npraw = np.array(raw_out.split())

    if npraw.size > 7: # if 7 or under, it means the gps got an error
        coorraw = str(npraw[7]).split(',')
        mode = coorraw[2][7] # mode 1 = no fix, mode 2 = 2D and mode 3 = 3D

        if mode != "1":
            nb_sat_raw = np.array(str(npraw[8]).split('used'))
            nb_sat = []
            nb_sats = 0
            for i in range(nb_sat_raw.size):
                if i != 0:
                    nb_sat.append(nb_sat_raw[i][2:6])
                    if nb_sat[i-1] == 'true':
                        nb_sats += 1

                lat = coorraw[5][6:]
                lon = coorraw[6][6:]
            log.write("Data correctly acquire\n")
            os.system("clear")
            print("Data correctly acquire")
        else:
            nb_sats = 0
            lat = 0
            lon = 0
            log.write("no fix\n")
            os.system("clear")
            print("no fix")
    else:
        lat = 999
        lon = 999
        nb_sats = 999
        os.system("clear")
        print("no")
        log.write("ERROR\n")
    
    #writer_data.writerow([time, lat, lon, nb_sats])
    data.write(str(time) + "," + str(lat) + "," + str(lon) + "," + str(nb_sats) + "\n")
    open("/var/www/html/data/live/gps_log.txt", 'w').close()
    data.close()