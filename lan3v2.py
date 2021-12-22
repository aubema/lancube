import smbus
import RPi.GPIO as GPIO
from datetime import datetime
import time
import csv
import os
import threading
import serial
import pynmea2

# Get I2C bus 
capteur = [0, 0, 0, 0, 0]
capteur[0] = smbus.SMBus(7)
capteur[1] = smbus.SMBus(3)
capteur[2] = smbus.SMBus(4)
capteur[3] = smbus.SMBus(5)
capteur[4] = smbus.SMBus(6)

# I2C Address of the device
TCS34725_DEFAULT_ADDRESS = 0x29

# TCS34725 Register Set
TCS34725_COMMAND_BIT = 0x80
TCS34725_REG_ENABLE = 0x00  # Enables states and interrupts
TCS34725_REG_ATIME = 0x01  # RGBC integration time
TCS34725_REG_WTIME = 0x03  # Wait time
TCS34725_REG_CONFIG = 0x0D  # Configuration register
TCS34725_REG_CONTROL = 0x0F  # Control register
TCS34725_REG_CDATAL = 0x14  # Clear/IR channel low data register
TCS34725_REG_CDATAH = 0x15  # Clear/IR channel high data register
TCS34725_REG_RDATAL = 0x16  # Red ADC low data register
TCS34725_REG_RDATAH = 0x17  # Red ADC high data register
TCS34725_REG_GDATAL = 0x18  # Green ADC low data register
TCS34725_REG_GDATAH = 0x19  # Green ADC high data register
TCS34725_REG_BDATAL = 0x1A  # Blue ADC low data register
TCS34725_REG_BDATAH = 0x1B  # Blue ADC high data register

# TCS34725 Enable Register Configuration
TCS34725_REG_ENABLE_SAI = 0x40  # Sleep After Interrupt
TCS34725_REG_ENABLE_AIEN = 0x10  # ALS Interrupt Enable
TCS34725_REG_ENABLE_WEN = 0x08  # Wait Enable
TCS34725_REG_ENABLE_AEN = 0x02  # ADC Enable
TCS34725_REG_ENABLE_PON = 0x01  # Power ON

# TCS34725 Time Register Configuration
TCS34725_REG_TIME_1 = 0xFF  # Atime = 2.4 ms, Cycles = 1
TCS34725_REG_TIME_2 = 0xFE  # Atime = 4.8 ms, Cycles = 2
TCS34725_REG_TIME_4 = 0xFC  # Atime = 9.6 ms, Cycles = 4
TCS34725_REG_TIME_8 = 0xF8  # Atime = 19.2 ms, Cycles = 8
TCS34725_REG_TIME_16 = 0xF0  # Atime = 38.4 ms, Cycles = 16
TCS34725_REG_TIME_32 = 0xE0  # Atime = 76.8 ms, Cycles = 32
TCS34725_REG_TIME_64 = 0xC0  # Atime = 153.6 ms, Cycles = 64
TCS34725_REG_TIME_128 = 0x80  # Atime = 307.2 ms, Cycles = 128
TCS34725_REG_TIME_256 = 0x00  # Atime = 614.4 ms, Cycles = 256

TIME_1 = 2.4
TIME_2 = 4.8
TIME_4 = 9.6
TIME_8 = 19.2
TIME_16 = 38.4
TIME_32 = 76.8
TIME_64 = 153.6
TIME_128 = 307.2
TIME_256 = 614.4

# saturation is related to the number of cycles
# SAT= cycle * 1024
SAT_1 = 1024
SAT_2 = 2048
SAT_4 = 4096
SAT_8 = 8192
SAT_16 = 16384
SAT_32 = 32768
SAT_64 = 65535
SAT_128 = 65535
SAT_256 = 65535

# TCS34725 Gain Configuration
TCS34725_REG_CONTROL_AGAIN_1 = 0x00  # 1x Gain
TCS34725_REG_CONTROL_AGAIN_4 = 0x01  # 4x Gain
TCS34725_REG_CONTROL_AGAIN_16 = 0x02  # 16x Gain
TCS34725_REG_CONTROL_AGAIN_60 = 0x03  # 60x Gain

# Def function


def enable_selection(sensor):
    """Select the ENABLE register configuration from the given provided values"""
    ENABLE_CONFIGURATION = (TCS34725_REG_ENABLE_AEN | TCS34725_REG_ENABLE_PON)
    sensor.write_byte_data(TCS34725_DEFAULT_ADDRESS, TCS34725_REG_ENABLE |
                           TCS34725_COMMAND_BIT, ENABLE_CONFIGURATION)


def time_selection(sensor, AT, WT):
    """Select the ATIME register configuration from the given provided values"""
    sensor.write_byte_data(TCS34725_DEFAULT_ADDRESS, TCS34725_REG_ATIME | TCS34725_COMMAND_BIT, AT)

    """Select the WTIME register configuration from the given provided values"""
    sensor.write_byte_data(TCS34725_DEFAULT_ADDRESS, TCS34725_REG_WTIME | TCS34725_COMMAND_BIT, WT)


def gain_selection(sensor, G):
    """Select the gain register configuration from the given provided values"""
    sensor.write_byte_data(TCS34725_DEFAULT_ADDRESS, TCS34725_REG_CONTROL | TCS34725_COMMAND_BIT, G)


def readluminance(sensor):
    """Read data back from TCS34725_REG_CDATAL(0x94), 8 bytes, with TCS34725_COMMAND_BIT, (0x80)
    cData LSB, cData MSB, Red LSB, Red MSB, Green LSB, Green MSB, Blue LSB, Blue MSB"""
    data = sensor.read_i2c_block_data(TCS34725_DEFAULT_ADDRESS,
                                      TCS34725_REG_CDATAL | TCS34725_COMMAND_BIT, 8)
    if (data[0] == 0):
       # possible saturation 
       data = sensor.read_i2c_block_data(TCS34725_DEFAULT_ADDRESS,
                                      TCS34725_REG_CDATAL | TCS34725_COMMAND_BIT, 8)
       
    # Convert the data
    cData = data[1] * 256 + data[0]
    red = data[3] * 256 + data[2]
    green = data[5] * 256 + data[4]
    blue = data[7] * 256 + data[6]
                               
                                              

    # Calculate luminance
    luminance = (-0.32466 * red) + (1.57837 * green) + (-0.73191 * blue)

    return {'c': cData, 'r': red, 'g': green, 'b': blue, 'l': luminance}

# Generate the name of the csv file with the current date


def name():
    today = str(datetime.now())
    year = today[0:4]
    month = today[5:7]
    day = today[8:10]

    name = "{}-{}-{}.csv".format(year, month, day)

    return name

# Get the date and return the desired values


def get_time():
    today = str(datetime.now())
    year = today[0:4]
    month = today[5:7]
    day = today[8:10]
    hour = int(today[11:13])
    minute = int(today[14:16])
    second = float("{:.2f}".format(float(today[17:25])))

    return {'year': year, 'month': month, 'day': day, 'hour': hour, 'min': minute, 'sec': second}

# Convert the hexadecimal gain in a digital one (used in flux() function)


def num_gain(current_gain):
    gain = 0

    if current_gain == TCS34725_REG_CONTROL_AGAIN_1:
        gain = 1
    elif current_gain == TCS34725_REG_CONTROL_AGAIN_4:
        gain = 4
    elif current_gain == TCS34725_REG_CONTROL_AGAIN_16:
        gain = 16
    elif current_gain == TCS34725_REG_CONTROL_AGAIN_60:
        gain = 60

    return gain

# Convert the hexadecimal integration time in a digital one (used in flux() function)


def num_acquisition_time(current_acquisition_time):
    acquisition_time = 0

    if current_acquisition_time == TCS34725_REG_TIME_1:
        acquisition_time = TIME_1
    elif current_acquisition_time == TCS34725_REG_TIME_2:
        acquisition_time = TIME_2
    elif current_acquisition_time == TCS34725_REG_TIME_4:
        acquisition_time = TIME_4
    elif current_acquisition_time == TCS34725_REG_TIME_8:
        acquisition_time = TIME_8
    elif current_acquisition_time == TCS34725_REG_TIME_16:
        acquisition_time = TIME_16        
    elif current_acquisition_time == TCS34725_REG_TIME_32:
        acquisition_time = TIME_32
    elif current_acquisition_time == TCS34725_REG_TIME_64:
        acquisition_time = TIME_64
    elif current_acquisition_time == TCS34725_REG_TIME_128:
        acquisition_time = TIME_128
    elif current_acquisition_time == TCS34725_REG_TIME_256:
        acquisition_time = TIME_256  

    return acquisition_time


# function that calculate de calibrated lux (and return "N/A" if there is a /0)
# the function is optimized for white LED lights
def clux(lux, Ga, AT):

	clux = 0

	if Ga != 0 and AT != 0:
		clux = lux / Ga / AT * 395
		clux = "{:.2f}".format(clux)

	else:
		clux = str(clux)
		clux = "N/A"

	return clux     

# Calculate MSI

def calc_msi(r, g, b, c):
    error = 0
    ir = (r + g + b - c)/2 
    r = r - ir
    g = g - ir
    b = b - ir
   
    if g > 0 :
        bg = b/g
        rg = r/g
        aa = 0.0769
        bb = 0.6023
        cc = -0.1736
        dd = -0.0489
        ee = 0.3098
        ff = 0.0257
        msi = aa + bb*bg + cc*rg + dd*bg*rg + ee*bg**2 + ff*rg**2   
        msi="{:.2f}".format(msi)     
    else:
        error = 1
        msi = 99

    return msi

# Calculate the color temperature


def colour_temperature(r, g, b, c):
    error = 0

#    if c != 0:
#        r = r/c*255
#        g = g/c*255
#        b = b/c*255
#
#    else:
#        error = 1
    ir = (r + g + b - c)/2
    r = r - ir
    g = g - ir
    b = b - ir

    X = (-0.14282)*r + 1.54924*g + (-0.95641)*b
    Y = (-0.32466)*r + 1.57837*g + (-0.73191)*b
    Z = (-0.68202)*r + 0.77073*g + 0.56332*b

    if X + Y + Z != 0:
        x = X/(X+Y+Z)
        y = Y/(X+Y+Z)
        n = (x-0.3320)/(0.1858-y)

    else:
        error = 1

    if error == 0:
        colour_temp = 449*(n**3) + 3525*(n**2) + 6823.3*n + 5520.33
        colour_temp = round(colour_temp)

    elif error == 1:
        colour_temp = "N/A"

    return colour_temp

# Write everything in the .csv file


def write_data(writer, sensor, year, month, day, hour, min, sec, lat, lon, alt, nSat, ga, acqt, temp, msii, lux, r, g, b, c, tail):

    writer.writerow(["S" + str(sensor), year, month, day, hour, min, sec, lat,
                    lon, alt, nSat, ga, acqt, temp, msii, lux, r, g, b, c, tail])


# Check if the data is over exposed, under exposed or correctly exposed and return the corresponding tail
def get_tail(red, green, blue, clear, current_acquisition_time):
    tail = "--"
    if current_acquisition_time == TCS34725_REG_TIME_256:
        sat = SAT_256
    elif current_acquisition_time == TCS34725_REG_TIME_128:
        sat = SAT_128
    elif current_acquisition_time == TCS34725_REG_TIME_64:
        sat = SAT_64
    elif current_acquisition_time == TCS34725_REG_TIME_32:
        sat = SAT_32
    elif current_acquisition_time == TCS34725_REG_TIME_16:
        sat = SAT_16
    elif current_acquisition_time == TCS34725_REG_TIME_8:
        sat = SAT_8
    elif current_acquisition_time == TCS34725_REG_TIME_4:
        sat = SAT_4            
    elif current_acquisition_time == TCS34725_REG_TIME_2:
        sat = SAT_2  
    else:
        sat = SAT_1  
    if (red >= sat or green >= sat or blue >= sat or clear >= sat):
        tail = "OE"
    elif (red+green+blue > 2*clear) or (red+green+blue < 0.7*clear):
        tail = "ER"
    elif clear <= 199:
        tail = "UE"
    else:
        tail = "OK"

    return tail

# Correction of gain and integration time


def correction(red, green, blue, clear, current_gain, current_acquisition_time, current_waiting_time):

    if current_acquisition_time == TCS34725_REG_TIME_256:
        sat = SAT_256
    elif current_acquisition_time == TCS34725_REG_TIME_128:
        sat = SAT_128
    elif current_acquisition_time == TCS34725_REG_TIME_64:
        sat = SAT_64
    elif current_acquisition_time == TCS34725_REG_TIME_32:
        sat = SAT_32
    elif current_acquisition_time == TCS34725_REG_TIME_16:
        sat = SAT_16
    elif current_acquisition_time == TCS34725_REG_TIME_8:
        sat = SAT_8
    elif current_acquisition_time == TCS34725_REG_TIME_4:
        sat = SAT_4            
    elif current_acquisition_time == TCS34725_REG_TIME_2:
        sat = SAT_2  
    else:
        sat = SAT_1  
    if (red >= sat or green >= sat or blue >= sat or clear >= sat):
        # more than 2 is infrared lamps
        print("ERROR - SENSOR SATURATION : Trying to correct the settings...")
        if current_gain == TCS34725_REG_CONTROL_AGAIN_60:
           current_gain = TCS34725_REG_CONTROL_AGAIN_16
        elif current_gain == TCS34725_REG_CONTROL_AGAIN_16:
           current_gain = TCS34725_REG_CONTROL_AGAIN_4
        elif current_gain == TCS34725_REG_CONTROL_AGAIN_4:
           current_gain = TCS34725_REG_CONTROL_AGAIN_1  
        elif current_acquisition_time == TCS34725_REG_TIME_256:
            current_acquisition_time = TCS34725_REG_TIME_128
        elif current_acquisition_time == TCS34725_REG_TIME_128:
            current_acquisition_time = TCS34725_REG_TIME_64
        elif current_acquisition_time == TCS34725_REG_TIME_64:
            current_acquisition_time = TCS34725_REG_TIME_32
        elif current_acquisition_time == TCS34725_REG_TIME_32:
            current_acquisition_time = TCS34725_REG_TIME_16
        elif current_acquisition_time == TCS34725_REG_TIME_16:
            current_acquisition_time = TCS34725_REG_TIME_8
        elif current_acquisition_time == TCS34725_REG_TIME_8:
            current_acquisition_time = TCS34725_REG_TIME_4
        elif current_acquisition_time == TCS34725_REG_TIME_4:
            current_acquisition_time = TCS34725_REG_TIME_2            
        elif current_acquisition_time == TCS34725_REG_TIME_2:
            current_acquisition_time = TCS34725_REG_TIME_1               
        else:
            print("There is just too much light...... :( ")
    elif clear <= 199:
        print("ERROR = SENSOR UNDEREXPOSED : Trying to correct the settings...")
        if current_gain == TCS34725_REG_CONTROL_AGAIN_1:
           current_gain = TCS34725_REG_CONTROL_AGAIN_4
        elif current_gain == TCS34725_REG_CONTROL_AGAIN_4:
           current_gain = TCS34725_REG_CONTROL_AGAIN_16
        elif current_gain == TCS34725_REG_CONTROL_AGAIN_16:
           current_gain = TCS34725_REG_CONTROL_AGAIN_60          
        elif current_acquisition_time == TCS34725_REG_TIME_1:
            current_acquisition_time = TCS34725_REG_TIME_2
        elif current_acquisition_time == TCS34725_REG_TIME_2:
            current_acquisition_time = TCS34725_REG_TIME_4
        elif current_acquisition_time == TCS34725_REG_TIME_4:
            current_acquisition_time = TCS34725_REG_TIME_8
        elif current_acquisition_time == TCS34725_REG_TIME_8:
            current_acquisition_time = TCS34725_REG_TIME_16
        elif current_acquisition_time == TCS34725_REG_TIME_16:
            current_acquisition_time = TCS34725_REG_TIME_32
        elif current_acquisition_time == TCS34725_REG_TIME_32:
            current_acquisition_time = TCS34725_REG_TIME_64
        elif current_acquisition_time == TCS34725_REG_TIME_64:
            current_acquisition_time = TCS34725_REG_TIME_128            
        elif current_acquisition_time == TCS34725_REG_TIME_128:
            current_acquisition_time = TCS34725_REG_TIME_256              
        else:
            print("There is just not enough light...... :( ")
    else:
        print("\b\bData have been correctly gathered")
    if current_acquisition_time == TCS34725_REG_TIME_1:
        current_waiting_time = TCS34725_REG_TIME_2
    elif current_acquisition_time == TCS34725_REG_TIME_2:
        current_waiting_time = TCS34725_REG_TIME_4
    elif current_acquisition_time == TCS34725_REG_TIME_4:
        current_waiting_time = TCS34725_REG_TIME_8
    elif current_acquisition_time == TCS34725_REG_TIME_8:
        current_waiting_time = TCS34725_REG_TIME_16
    elif current_acquisition_time == TCS34725_REG_TIME_16:
        current_waiting_time = TCS34725_REG_TIME_32
    elif current_acquisition_time == TCS34725_REG_TIME_32:
        current_waiting_time = TCS34725_REG_TIME_64
    elif current_acquisition_time == TCS34725_REG_TIME_64:
        current_waiting_time = TCS34725_REG_TIME_128            
    elif current_acquisition_time == TCS34725_REG_TIME_128:
        current_waiting_time = TCS34725_REG_TIME_256
    else:
        current_waiting_time = TCS34725_REG_TIME_256 
    return {'c_g': current_gain, 'c_at': current_acquisition_time, 'c_wt': current_waiting_time}


def largest(arr):
    num = [0, 0, 0, 0, 0]

    num[0] = num_acquisition_time(arr[0])
    num[1] = num_acquisition_time(arr[1])
    num[2] = num_acquisition_time(arr[2])
    num[3] = num_acquisition_time(arr[3])
    num[4] = num_acquisition_time(arr[4])
# Initialize maximum element

    max = num[0]

# Traverse array elements from second and compare every element with current max
    for i in range(5):
        if num[i] > max:
            max = num[i]
    return max


# setup GPIO end pins for LED
redPin = 38
greenPin = 40
bluePin = 37

# general GPIO setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# button GPIO setup
GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(19, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# led GPIO setup
GPIO.setup(redPin, GPIO.OUT)
GPIO.setup(greenPin, GPIO.OUT)
GPIO.setup(bluePin, GPIO.OUT)

# UPS gpio setup
# GPIO.setup(11, GPIO.IN)

# All colors LED functions


def blink(pin):
    GPIO.output(pin, GPIO.HIGH)


def turnOff(pin):
    GPIO.output(pin, GPIO.LOW)


def redOn():
    blink(redPin)


def redOff():
    turnOff(redPin)


def greenOn():
    blink(greenPin)


def greenOff():
    turnOff(greenPin)


def blueOn():
    blink(bluePin)


def blueOff():
    turnOff(bluePin)


def yellowOn():
    blink(redPin)
    blink(greenPin)


def yellowOff():
    turnOff(redPin)
    turnOff(greenPin)


def whiteOn():
    blink(redPin)
    blink(greenPin)
    blink(bluePin)


def whiteOff():
    turnOff(redPin)
    turnOff(greenPin)
    turnOff(bluePin)

# UPS hat shutting down after 60 sec without power


def ups():
    global end

    while end == 0:
        if GPIO.input(11) == 1:
            print("WARNING, Lancube has no more power, shutting down in 60 sec if power is not recovered")

            for i in range(60):
                if GPIO.input(11) == 1 and i == 59:
                    end = 1
                elif GPIO.input(11) == 0:
                    print("Power correctly recovered, lancube not shutting down")
                    break
                else:
                    print("waiting for power recovery...")
                    time.sleep(1)

# Get gps position


def getPositionData():
    global end
    global lat
    global lon
    global alt
    global nbSats
    global times

    SERIAL_PORT = "/dev/ttyACM0"
    try:
        gps = serial.Serial(SERIAL_PORT, baudrate=9600, timeout=0.5)
    except serial.SerialException:
        print("No GPS module found...")

    while end == 0:
        try:
            data = gps.readline()
            message = str(data[0:6])
            message = message[2:8]
            message_all = str(data)
            message_all = message_all[2:-1]

            if (message == "$GPGGA"):
                # GPGGA = Global Positioning System Fix Data
                # Reading the GPS fix data is an alternative approach that also works
                today = str(datetime.now())
                hour = int(today[11:13])
                minute = int(today[14:16])
                second = float(today[17:])
                times[0] = times[1]
                times[1] = hour*3600 + minute*60 + second
                data = str(data)
                data = data[2:-5]

                parts = pynmea2.parse(data)

                if int(parts.gps_qual) == 0:
                    # Equal to 0 = No gps fix...
                    print("No gps fix")
                    lat[0] = 0
                    lon[0] = 0
                    alt[0] = 0
                    lat[1] = 0
                    lon[1] = 0
                    alt[1] = 0
                    nbSats = 0
                else:
                    # Get the position data that was transmitted with the GPGGA message
                    lat[0] = lat[1]
                    lon[0] = lon[1]
                    alt[0] = alt[1]

                    lat[1] = float("{:.6f}".format(parts.latitude))
                    lon[1] = float("{:.6f}".format(parts.longitude))
                    alt[1] = float("{:.6f}".format(parts.altitude))
                    nbSats = int(parts.num_sats)

            else:
                # Handle other NMEA messages and unsupported strings
                pass
        except KeyboardInterrupt:
            gps.close()
            print("Application closed!")
        except:
            lat[0] = 0
            lon[0] = 0
            alt[0] = 0
            lat[1] = 0
            lon[1] = 0
            alt[1] = 0
            nbSats = 0

            time.sleep(0.9)
            try:
                if SERIAL_PORT == "/dev/ttyACM0":
                    SERIAL_PORT = "/dev/ttyACM1"
                elif SERIAL_PORT == "/dev/ttyACM1":
                    SERIAL_PORT = "/dev/ttyACM0"

                gps = serial.Serial(SERIAL_PORT, baudrate=9600, timeout=0.5)

            except serial.SerialException:
                print("No GPS module found...")


# initialisation
# LED
whiteOff()
yellowOn()

# name of the file
name1 = name()
name1_update = name1

# GS = Gain Sensor (lowest possible)
GS = [0, 0, 0, 0, 0]
GS[0] = TCS34725_REG_CONTROL_AGAIN_16
GS[1] = TCS34725_REG_CONTROL_AGAIN_16
GS[2] = TCS34725_REG_CONTROL_AGAIN_16
GS[3] = TCS34725_REG_CONTROL_AGAIN_16
GS[4] = TCS34725_REG_CONTROL_AGAIN_16

# ATS = Acquisition Time Sensor (fastest possible)
ATS = [0, 0, 0, 0, 0]
ATS[0] = TCS34725_REG_TIME_128
ATS[1] = TCS34725_REG_TIME_128
ATS[2] = TCS34725_REG_TIME_128
ATS[3] = TCS34725_REG_TIME_128
ATS[4] = TCS34725_REG_TIME_128

# WTS = Wainting Time Sensor (one step over ATS)
WTS = [0, 0, 0, 0, 0]
WTS[0] = TCS34725_REG_TIME_128
WTS[1] = TCS34725_REG_TIME_128
WTS[2] = TCS34725_REG_TIME_128
WTS[3] = TCS34725_REG_TIME_128
WTS[4] = TCS34725_REG_TIME_128

data = open('/var/www/html/data/' + name1, 'a')
writer = csv.writer(data)
if os.stat('/var/www/html/data/' + name1).st_size <= 0:
    writer.writerow(["Sensor", "Year", "Month", "Day", "Hour", "Minute", "Second", "Latitude", "Longitude", "Altitude", "NumberSatellites",
                    "Gain", "AcquisitionTime(ms)", "ColorTemperature(k)", "MSI", "lux", "Red", "Green", "Blue", "Clear", "Flag"])

# Initialize values of the variables
tail = ["--", "--", "--", "--", "--"]
button_status = 0
end = 0
i = 0
running = True
today = str(datetime.now())
year = today[0:4]
month = today[5:7]
day = today[8:10]
lat = [0, 0]
lon = [0, 0]
alt = [0, 0]
nbSats = 0
times = [0, 0]

# Gps thread initialisation
tGps = threading.Thread(target=getPositionData, name="Gps thread")
tGps.start()
# tUps = threading.Thread(target=ups, name="Ups thread")
# tUps.start()

# Main loop
while end == 0:
    # If the day change, create a new file with the correct name
    if name1_update != name1:
        data.close()
        data = open('/var/www/html/data/' + name1_update, 'a')
        writer = csv.writer(data)
        writer.writerow(["Sensor", "Year", "Month", "Day", "Hour", "Minute", "Second", "Latitude", "Longitude", "Altitude", "NumberSatellites",
                         "Gain", "AcquisitionTime(ms)", "ColorTemperature(k)", "MSI", "lux", "Red", "Green", "Blue", "Clear", "Flag"])
        name1 = name1_update

    name1_update = name()

    # Look for button status
    if GPIO.input(21) == 1:
        button_status = 1
    elif GPIO.input(19) == 1:
        button_status = 2
    else:
        button_status = 0

    if button_status == 1:
        print("-----------------------------data ", i+1, "------------------------------------")

        for a in range(5):
            enable_selection(capteur[a])
            time_selection(capteur[a], ATS[a], WTS[a])
            gain_selection(capteur[a], GS[a])
            lum = readluminance(capteur[a])
            ntry = 1
            while (( lum['r']+lum['g']+lum['b'] > 1.3*lum['c']) or (lum['r']+lum['g']+lum['b'] < 0.7*lum['c']) or (lum['l'] < 0)) and ( ntry < 999):
               lum = readluminance(capteur[a])
               ntry += 1
            time_str = get_time()
            gain = num_gain(GS[a])
            acqt = num_acquisition_time(ATS[a])
            lux = clux(lum['l'] , gain, acqt)
            temp = colour_temperature(lum['r'], lum['g'], lum['b'], lum['c'])
            msi_index = calc_msi(lum['r'], lum['g'], lum['b'], lum['c'])
            tail[a] = get_tail(lum['r'], lum['g'], lum['b'], lum['c'], ATS[a])
            print("Corrected lux=", lux)
            # Used for gps
            today = str(datetime.now())
            hour = int(today[11:13])
            minute = int(today[14:16])
            second = float(today[17:])
            time_sec = hour*3600 + \
                minute*60 + second

            # Interpolation of the position
            if (times[1]-times[0]) != 0 and lat[0] != 0 and lat[1] != 0 and lon[0] != 0 and lon[1] != 0:
                future_lat = float("{:.6f}".format(
                    (lat[1]-lat[0])/(times[1]-times[0])*(time_sec-times[1])+lat[1]))
                future_lon = float("{:.6f}".format(
                    (lon[1]-lon[0])/(times[1]-times[0])*(time_sec-times[1])+lon[1]))
                future_alt = float("{:.1f}".format(
                    (alt[1]-alt[0])/(times[1]-times[0])*(time_sec-times[1])+alt[1]))
            else:
                future_lat = 0
                future_lon = 0
                future_alt = 0

            # write the line of the sensor 1 in the csv file
            write_data(writer, a+1, time_str['year'], time_str['month'], time_str['day'], time_str['hour'], time_str['min'],
                       time_str['sec'], future_lat, future_lon, future_alt, nbSats, gain, acqt, temp, msi_index, lux, lum['r'], lum['g'], lum['b'], lum['c'], tail[a])

            # Correction of the gain and integration time for sensor 1
            corr = correction(lum['r'], lum['g'], lum['b'], lum['c'], GS[a], ATS[a], WTS[a])
            GS[a] = corr['c_g']
            ATS[a] = corr['c_at']
            WTS[a] = corr['c_wt']

        if tail[0] != "OK" or tail[1] != "OK" or tail[2] != "OK" or tail[3] != "OK" or tail[4] != "OK":
            whiteOff()
            if nbSats > 3:
               blueOn()
            else:
               blueOn()
               redOn()
#            time.sleep(largest(ATS)/1000)
            
        elif tail[0] == "OK" and tail[1] == "OK" and tail[2] == "OK" and tail[3] == "OK" and tail[4] == "OK" and nbSats <= 3:
            whiteOff()
            yellowOn()
#            time.sleep(0.2)
        elif tail[0] == "OK" and tail[1] == "OK" and tail[2] == "OK" and tail[3] == "OK" and tail[4] == "OK" and nbSats > 3:
            whiteOff()
            greenOn()
#            time.sleep(0.2)
        tslp=largest(ATS)/1000
        time.sleep(tslp)        
        if tslp < 0.2:
           tslp = 0.2-tslp
           time.sleep(tslp)
        i = i + 1

    elif button_status == 0:
        print("IDLE...")
        whiteOff()
        yellowOn()
        time.sleep(0.5)
        whiteOff()
        time.sleep(0.5)

    elif button_status == 2:
        end = 1

# Shutdown sequence
print("Shutdown")
whiteOff()
redOn()
data.close()
# Waiting for the treads to end before cleaning GPIO
time.sleep(1)
GPIO.cleanup()
os.system("sudo shutdown -h now")
