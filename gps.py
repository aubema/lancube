import serial
import pynmea2
from time import gmtime, strftime

SERIAL_PORT = "/dev/ttyACM0"
running = True

today = strftime("%Y-%m-%d %H:%M:%S", gmtime())
year = today[0:4]
month = today[5:7]
day = today[8:10]

data = open("/var/www/html/data/gps.txt", 'w', newline='').close()


def getPositionData(gps):

    data = gps.readline()
    message = str(data[0:6])
    message = message[2:8]
    message_all = str(data)
    message_all = message_all[2:-1]

    if (message == "$GPGGA"):
        # GPGGA = Global Positioning System Fix Data
        # Reading the GPS fix data is an alternative approach that also works
        today = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        hour = int(today[11:13])
        minute = int(today[14:16])
        second = float(today[17:])
        times = hour*3600 + minute*60 + second
        data = str(data)
        data = data[2:-5]

        parts = pynmea2.parse(data)

        if int(parts.gps_qual) != 1:
            # Different from 1 = No gps fix...
            print("No gps fix")
            lat = 0
            lon = 0
            alt = 0
            nbSats = 0
        else:
            # Get the position data that was transmitted with the GPGGA message
            lat = parts.latitude
            lon = parts.longitude
            alt = parts.altitude
            nbSats = parts.num_sats
            print(parts.longitude)

        data = open("/var/www/html/data/gps.txt", 'a', newline='')
        data.write(str(times) + "," + str(lat)[0:10] + "," + str(lon)[0:10] +
                   "," + str(alt) + "," + str(nbSats) + "\n")
        data.close()
    else:
        # Handle other NMEA messages and unsupported strings
        pass


print("Application started!")
gps = serial.Serial(SERIAL_PORT, baudrate=9600, timeout=0.5)
print("serial open")

while running:
    try:
        getPositionData(gps)
    except KeyboardInterrupt:
        running = False
        gps.close()
        print("Application closed!")
    except:
        print("ERROR")
        lat = 99
        lon = 99
        alt = 99
        nbSats = 99
        data.write(str(times) + "," + str(lat)[0:10] + "," + str(lon)[0:10] +
                   "," + str(alt) + "," + str(nbSats) + "\n")
