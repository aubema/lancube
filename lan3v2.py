# Distributed with a free-will license.
# Use it any way you want, profit or free, provided it fits in the licenses of its associated works.
# TCS34725
# This code is designed to work with the TCS34725_I2CS I2C Mini Module available from ControlEverything.com.
# https://www.controleverything.com/products
# NT
 
import smbus
import RPi.GPIO as GPIO
import numpy as np
import gpsd
from datetime import datetime
import time
import csv
import os
from subprocess import check_call

# Get I2C bus
capteur1 = smbus.SMBus(3)
capteur2 = smbus.SMBus(4)
capteur3 = smbus.SMBus(5)
capteur4 = smbus.SMBus(6)
capteur5 = smbus.SMBus(7)
 
# I2C Address of the device
TCS34725_DEFAULT_ADDRESS = 0x29
 
# TCS34725 Register Set
TCS34725_COMMAND_BIT = 0x80
TCS34725_REG_ENABLE = 0x00 # Enables states and interrupts
TCS34725_REG_ATIME = 0x01 # RGBC integration time
TCS34725_REG_WTIME = 0x03 # Wait time
TCS34725_REG_CONFIG = 0x0D # Configuration register
TCS34725_REG_CONTROL = 0x0F # Control register
TCS34725_REG_CDATAL = 0x14 # Clear/IR channel low data register
TCS34725_REG_CDATAH = 0x15 # Clear/IR channel high data register
TCS34725_REG_RDATAL = 0x16 # Red ADC low data register
TCS34725_REG_RDATAH = 0x17 # Red ADC high data register
TCS34725_REG_GDATAL = 0x18 # Green ADC low data register
TCS34725_REG_GDATAH = 0x19 # Green ADC high data register
TCS34725_REG_BDATAL = 0x1A # Blue ADC low data register
TCS34725_REG_BDATAH = 0x1B # Blue ADC high data register
 
# TCS34725 Enable Register Configuration
TCS34725_REG_ENABLE_SAI = 0x40 # Sleep After Interrupt
TCS34725_REG_ENABLE_AIEN = 0x10 # ALS Interrupt Enable
TCS34725_REG_ENABLE_WEN = 0x08 # Wait Enable
TCS34725_REG_ENABLE_AEN = 0x02 # ADC Enable
TCS34725_REG_ENABLE_PON = 0x01 # Power ON
 
# TCS34725 Time Register Configuration
TCS34725_REG_ATIME_2_4 = 0xFF # Atime = 2.4 ms, Cycles = 1
TCS34725_REG_ATIME_9_6 = 0xFC # Atime = 9.6 ms, Cycles = 4
TCS34725_REG_ATIME_38_4 = 0xF0 # Atime = 38.4 ms, Cycles = 16
TCS34725_REG_ATIME_153_6 = 0xC0 # Atime = 153.6 ms, Cycles = 64
TCS34725_REG_ATIME_614_4 = 0x00 # Atime = 614.4 ms, Cycles = 256
TCS34725_REG_WTIME_4_8 = 0xFE # Wtime = 4.8 ms, Cycles 2
TCS34725_REG_WTIME_12 = 0xFB # Wtime = 12 ms, Cycles 5
TCS34725_REG_WTIME_40_8 = 0xEF # Wtime = 40.8 ms, Cycles 17
TCS34725_REG_WTIME_156 = 0xD3 # Wtime = 4156 ms , Cycles 65
TCS34725_REG_WTIME_614_4 = 0x00 # Wtime = 614.4 ms Cycles 256
 
# TCS34725 Gain Configuration
TCS34725_REG_CONTROL_AGAIN_1 = 0x00 # 1x Gain
TCS34725_REG_CONTROL_AGAIN_4 = 0x01 # 4x Gain
TCS34725_REG_CONTROL_AGAIN_16 = 0x02 # 16x Gain
TCS34725_REG_CONTROL_AGAIN_60 = 0x03 # 60x Gain

#def function
def enable_selection(sensor):
	"""Select the ENABLE register configuration from the given provided values"""
	ENABLE_CONFIGURATION = (TCS34725_REG_ENABLE_AEN | TCS34725_REG_ENABLE_PON)
	sensor.write_byte_data(TCS34725_DEFAULT_ADDRESS, TCS34725_REG_ENABLE | TCS34725_COMMAND_BIT, ENABLE_CONFIGURATION)

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
	data = sensor.read_i2c_block_data(TCS34725_DEFAULT_ADDRESS, TCS34725_REG_CDATAL | TCS34725_COMMAND_BIT, 8)

	# Convert the data
	cData = data[1] * 256 + data[0]
	red = data[3] * 256 + data[2]
	green = data[5] * 256 + data[4]
	blue = data[7] * 256 + data[6]

	# Calculate luminance
	luminance = (-0.32466 * red) + (1.57837 * green) + (-0.73191 * blue)

	return {'c' : cData, 'r' : red, 'g' : green, 'b' : blue, 'l' : luminance}

# function that generate the name of the csv file with the current date
def name():
	today = str(datetime.today())
	year = today[0:4]
	month = today[5:7]
	day = today[8:10]

	name = "{}-{}-{}.csv".format(year, month, day)

	return name

# function that get the date and return the desired values
def get_time():
	today = str(datetime.today())
	year = today[0:4]
	month = today[5:7]
	day = today[8:10]
	hour = today[11:13]
	minute = today[14:16]
	second = today[17:19]
	
	return {'year' : year, 'month' : month, 'day' : day, 'hour' : hour, 'min' : minute, 'sec' : second} 

# function that get the current location and return each desired values
def get_location():
	packet = gpsd.get_current()
	
	lat = packet.lat
	lon = packet.lon
	alt = packet.alt
	nSat = packet.sats_valid

	return {'lat' : lat, 'lon' : lon, 'alt' : alt, 'nSat' : nSat}

# function that convert the hexadecimal gain in a digital one (used in flux() function)
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

# function that convert the hexadecimal integration time in a digital one (used in flux() function)
def num_acquisition_time(current_acquisition_time):
	acquisition_time = 0

	if current_acquisition_time == TCS34725_REG_ATIME_2_4:
		acquisition_time = 2.4
	elif current_acquisition_time == TCS34725_REG_ATIME_9_6:
		acquisition_time = 9.6
	elif current_acquisition_time == TCS34725_REG_ATIME_38_4:
		acquisition_time = 38.4
	elif current_acquisition_time == TCS34725_REG_ATIME_153_6:
		acquisition_time = 153.6
	elif current_acquisition_time == TCS34725_REG_ATIME_614_4:
		acquisition_time = 614.4

	return acquisition_time

# function that calculate the color temperature (and return N/A if there is a /0)
def colour_temperature(r, g, b, c):
	error = 0

	if c != 0:
		r = r/c*255
		g = g/c*255
		b = b/c*255

	else:
		error = 1


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

# function that calculate de flux (and return "N/A" if there is a /0)
def flux(lux, Ga, AT):

	flux = 0

	if Ga != 0 and AT != 0:
		flux = lux / Ga / AT * 481
		flux = "{:.2f}".format(flux)

	else:
		flux = str(flux)
		flux = "N/A"

	return flux

# function that write everything in the .csv file
def write_data(writer, sensor, year, month, day, hour, min, sec, lat, lon, alt, nSat, ga, acqt, temp, flux, lux, r, g, b, c, tail):
	
	
	writer.writerow([sensor, year, month, day, hour, min, sec, lat, lon, alt, nSat, ga, acqt, temp, flux, lux, r, g, b, c, tail])
	

# function that verify if the data is over exposed, under exposed or correctly exposed and return the corresponding tail
def tail(red, green, blue, clear):
	tail = "--"

	if ( red >= 40000 or green >= 40000 or blue >= 40000 or clear >= 40000  ) or ( red == green and red == blue and red > 100 ):
		tail = "OE"
	elif red <= 99 or green <= 99 or blue <= 99 or clear <= 99:
		tail = "UE"
	else:
		tail = "OK"

	return tail

# function that correct the gain and the integration time for a specific red, green, blue and clear color
def correction(red, green, blue, clear, current_gain, current_acquisition_time, current_waiting_time):

	if ( red >= 40000 or green >= 40000 or blue >= 40000 or clear >= 40000  ) or ( red == green and red == blue and red > 100 ):
		print("ERROR - SENSOR SATURATION : Trying to correct the settings...")

		if current_acquisition_time == TCS34725_REG_ATIME_614_4:
			current_acquisition_time = TCS34725_REG_ATIME_153_6
		elif current_acquisition_time == TCS34725_REG_ATIME_153_6:
			current_acquisition_time = TCS34725_REG_ATIME_38_4
		elif current_acquisition_time == TCS34725_REG_ATIME_38_4:
			current_acquisition_time = TCS34725_REG_ATIME_9_6
		elif current_acquisition_time == TCS34725_REG_ATIME_9_6:
			current_acquisition_time = TCS34725_REG_ATIME_2_4
		elif current_gain == TCS34725_REG_CONTROL_AGAIN_60:
			current_gain = TCS34725_REG_CONTROL_AGAIN_16
		elif current_gain == TCS34725_REG_CONTROL_AGAIN_16:
			current_gain = TCS34725_REG_CONTROL_AGAIN_4
		elif current_gain == TCS34725_REG_CONTROL_AGAIN_4:
			current_gain = TCS34725_REG_CONTROL_AGAIN_1
		else:
			print("There is just too much light...... :( ")

	elif red <= 99 or green <= 99 or blue <= 99 or clear <= 99:
		print("ERROR = SENSOR UNDERSATURATION : Trying to correct the settings...")

		if current_gain == TCS34725_REG_CONTROL_AGAIN_1:
			current_gain = TCS34725_REG_CONTROL_AGAIN_4
		elif current_gain == TCS34725_REG_CONTROL_AGAIN_4:
			current_gain = TCS34725_REG_CONTROL_AGAIN_16
		elif current_gain == TCS34725_REG_CONTROL_AGAIN_16:
			current_gain = TCS34725_REG_CONTROL_AGAIN_60
		elif current_acquisition_time == TCS34725_REG_ATIME_2_4:
			current_acquisition_time = TCS34725_REG_ATIME_9_6
		elif current_acquisition_time == TCS34725_REG_ATIME_9_6:
			current_acquisition_time = TCS34725_REG_ATIME_38_4
		elif current_acquisition_time == TCS34725_REG_ATIME_38_4:
			current_acquisition_time = TCS34725_REG_ATIME_153_6
		elif current_acquisition_time == TCS34725_REG_ATIME_153_6:
			current_acquisition_time = TCS34725_REG_ATIME_614_4
		else:
			print("There is just not enaugh light...... :( ")

	else:
		print("\b\bData have been correctly gathered")

	if current_acquisition_time == TCS34725_REG_ATIME_2_4:
		current_waiting_time = TCS34725_REG_WTIME_4_8
	elif current_acquisition_time == TCS34725_REG_ATIME_9_6:
		current_waiting_time = TCS34725_REG_WTIME_12
	elif current_acquisition_time == TCS34725_REG_ATIME_38_4:
		current_waiting_time = TCS34725_REG_WTIME_40_8
	elif current_acquisition_time == TCS34725_REG_ATIME_153_6:
		current_waiting_time = TCS34725_REG_WTIME_156
	elif current_acquisition_time == TCS34725_REG_ATIME_614_4:
		current_waiting_time = TCS34725_REG_WTIME_614_4

	return {'c_g' : current_gain, 'c_at' : current_acquisition_time, 'c_wt' : current_waiting_time}

# setup GPIO end pins for LED
redPin = 3
greenPin = 5
bluePin = 7

# general GPIO setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# button GPIO septup
GPIO.setup(3, GPIO.OUT)
GPIO.setup(5, GPIO.OUT)
GPIO.setup(7, GPIO.OUT)

# all colors LED functions
def blink(pin):
    GPIO.output(pin, GPIO.HIGH)
    
def turnOff(pin):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT)
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

def cyanOn():
    blink(greenPin)
    blink(bluePin)

def cyanOff():
    turnOff(greenPin)
    turnOff(bluePin)

def magentaOn():
    blink(redPin)
    blink(bluePin)

def magentaOff():
    turnOff(redPin)
    turnOff(bluePin)

def whiteOn():
    blink(redPin)
    blink(greenPin)
    blink(bluePin)

def whiteOff():
    turnOff(redPin)
    turnOff(greenPin)
    turnOff(bluePin)

#initialisation
#LED
whiteOff()
yellowOn()

#name of the file
name1 = name()

#connect GPS
gpsd.connect()

#TEMPORARY
data = open('/home/pi/Capteurs/' + name1, 'w', newline='')
writer = csv.writer(data)
writer.writerow(["Sensor", "Year", "Month", "Day", "Hour", "Minute", "Second", "Latitude", "Longitude", "Altitude", "Number of effective Satellites", "Gain", "Acquisition time (ms)", "Color Temperature (k)", "Flux", "lux", "Red", "Green", "Blue", "Clear", "Flag"])

#GS = Gain Sensor (lowest possible)
GS1 = TCS34725_REG_CONTROL_AGAIN_1
GS2 = TCS34725_REG_CONTROL_AGAIN_1
GS3 = TCS34725_REG_CONTROL_AGAIN_1
GS4 = TCS34725_REG_CONTROL_AGAIN_1
GS5 = TCS34725_REG_CONTROL_AGAIN_1

#ATS = Acquisition Time Sensor (fastest possible)
ATS1 = TCS34725_REG_ATIME_2_4
ATS2 = TCS34725_REG_ATIME_2_4
ATS3 = TCS34725_REG_ATIME_2_4
ATS4 = TCS34725_REG_ATIME_2_4
ATS5 = TCS34725_REG_ATIME_2_4

#WTS = Wainting Time Sensor (one step over ATS)
WTS1 = TCS34725_REG_WTIME_4_8
WTS2 = TCS34725_REG_WTIME_4_8
WTS3 = TCS34725_REG_WTIME_4_8
WTS4 = TCS34725_REG_WTIME_4_8
WTS5 = TCS34725_REG_WTIME_4_8

#TEMPORARY (suite)
data.close()
data = open('/home/pi/Capteurs/' + name1, 'a')
writer = csv.writer(data)

#initial value variables
tail1 = ["--", "--", "--", "--", "--"]
button_status = 0
end = 0
i = 0

# button GPIO setup
GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(19, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Main loop
while end == 0:

	if GPIO.input(21) == 1:
		button_status = 1
	elif GPIO.input(19) == 1:
		button_status = 2
	else:
		button_status = 0

	if button_status == 1:
		print("-----------------------------data ", i+1, "------------------------------------")

		enable_selection(capteur1)
		time_selection(capteur1, ATS1, WTS1)
		gain_selection(capteur1, GS1)

		lum = readluminance(capteur1)
		time1 = get_time()
		gps = get_location()
		gain = num_gain(GS1)
		acqt = num_acquisition_time(ATS1)
		temp = colour_temperature(lum['r'], lum['g'], lum['b'], lum['c'])
		tail1[0] = tail(lum['r'], lum['g'], lum['b'], lum['c'])
		flux1 = flux(lum['l'], num_gain(GS1), num_acquisition_time(ATS1))
		lux = "{:.2f}".format(lum['l'])

		# write the line of the sensor 1 in the csv file
		
		write_data(writer, "S1", time1['year'], time1['month'], time1['day'], time1['hour'], time1['min'], time1['sec'], gps['lat'], gps['lon'], gps['alt'], gps['nSat'], gain, acqt, temp, flux1, lux, lum['r'], lum['g'], lum['b'], lum['c'], tail1[0])

		# Correction of the gain and integration time for sensor 1
		corr = correction(lum['r'], lum['g'], lum['b'], lum['c'], GS1, ATS1, WTS1)
		GS1 = corr['c_g']
		ATS1 = corr['c_at']
		WTS1 = corr['c_wt']

		# same thing for sensor 2
		enable_selection(capteur2)
		time_selection(capteur2, ATS2, WTS2)
		gain_selection(capteur2, GS2)

		lum = readluminance(capteur2)
		time1 = get_time()
		gps = get_location()
		gain = num_gain(GS2)
		acqt = num_acquisition_time(ATS2)
		temp = colour_temperature(lum['r'], lum['g'], lum['b'], lum['c'])
		tail1[1] = tail(lum['r'], lum['g'], lum['b'], lum['c'])
		flux1 = flux(lum['l'], num_gain(GS2), num_acquisition_time(ATS2))
		lux = "{:.2f}".format(lum['l'])

		# write the line of the sensor 2 in the csv file
		
		write_data(writer, "S2", time1['year'], time1['month'], time1['day'], time1['hour'], time1['min'], time1['sec'], gps['lat'], gps['lon'], gps['alt'], gps['nSat'], gain, acqt, temp, flux1, lux, lum['r'], lum['g'], lum['b'], lum['c'], tail1[0])

		# Correction of the gain and integration time for sensor 2
		corr = correction(lum['r'], lum['g'], lum['b'], lum['c'], GS2, ATS2, WTS2)
		GS2 = corr['c_g']
		ATS2 = corr['c_at']
		WTS2 = corr['c_wt']

		# same thing for sensor 3
		enable_selection(capteur3)
		time_selection(capteur3, ATS3, WTS3)
		gain_selection(capteur3, GS3)

		lum = readluminance(capteur3)
		time1 = get_time()
		gps = get_location()
		gain = num_gain(GS3)
		acqt = num_acquisition_time(ATS3)
		temp = colour_temperature(lum['r'], lum['g'], lum['b'], lum['c'])
		tail1[2] = tail(lum['r'], lum['g'], lum['b'], lum['c'])
		flux1 = flux(lum['l'], num_gain(GS3), num_acquisition_time(ATS3))
		lux = "{:.2f}".format(lum['l'])

		# write the line of the sensor 3 in the csv file
		
		write_data(writer, "S3", time1['year'], time1['month'], time1['day'], time1['hour'], time1['min'], time1['sec'], gps['lat'], gps['lon'], gps['alt'], gps['nSat'], gain, acqt, temp, flux1, lux, lum['r'], lum['g'], lum['b'], lum['c'], tail1[0])

		# Correction of the gain and integration time for sensor 3
		corr = correction(lum['r'], lum['g'], lum['b'], lum['c'], GS1, ATS3, WTS3)
		GS3 = corr['c_g']
		ATS3 = corr['c_at']
		WTS3 = corr['c_wt']

		# same thing for sensor 4
		enable_selection(capteur4)
		time_selection(capteur4, ATS4, WTS4)
		gain_selection(capteur4, GS4)

		lum = readluminance(capteur4)
		time1 = get_time()
		gps = get_location()
		gain = num_gain(GS4)
		acqt = num_acquisition_time(ATS4)
		temp = colour_temperature(lum['r'], lum['g'], lum['b'], lum['c'])
		tail1[3] = tail(lum['r'], lum['g'], lum['b'], lum['c'])
		flux1 = flux(lum['l'], num_gain(GS4), num_acquisition_time(ATS4))
		lux = "{:.2f}".format(lum['l'])

		# write the line of the sensor 4 in the csv file
		
		write_data(writer, "S4", time1['year'], time1['month'], time1['day'], time1['hour'], time1['min'], time1['sec'], gps['lat'], gps['lon'], gps['alt'], gps['nSat'], gain, acqt, temp, flux1, lux, lum['r'], lum['g'], lum['b'], lum['c'], tail1[0])

		# Correction of the gain and integration time for sensor 4
		corr = correction(lum['r'], lum['g'], lum['b'], lum['c'], GS4, ATS4, WTS4)
		GS4 = corr['c_g']
		ATS4 = corr['c_at']
		WTS4 = corr['c_wt']

		# same thing for sensor 5
		enable_selection(capteur5)
		time_selection(capteur5, ATS5, WTS5)
		gain_selection(capteur5, GS5)

		lum = readluminance(capteur5)
		time1 = get_time()
		gps = get_location()
		gain = num_gain(GS5)
		acqt = num_acquisition_time(ATS5)
		temp = colour_temperature(lum['r'], lum['g'], lum['b'], lum['c'])
		tail1[4] = tail(lum['r'], lum['g'], lum['b'], lum['c'])
		flux1 = flux(lum['l'], num_gain(GS5), num_acquisition_time(ATS5))
		lux = "{:.2f}".format(lum['l'])

		# write the line of the sensor 5 in the csv file
		
		write_data(writer, "S5", time1['year'], time1['month'], time1['day'], time1['hour'], time1['min'], time1['sec'], gps['lat'], gps['lon'], gps['alt'], gps['nSat'], gain, acqt, temp, flux1, lux, lum['r'], lum['g'], lum['b'], lum['c'], tail1[0])

		# Correction of the gain and integration time for sensor 5
		corr = correction(lum['r'], lum['g'], lum['b'], lum['c'], GS5, ATS5, WTS5)
		GS5 = corr['c_g']
		ATS5 = corr['c_at']
		WTS5 = corr['c_wt']

		if tail1[0] != "OK" or tail1[1] != "OK" or tail1[2] != "OK" or tail1[3] != "OK" or tail1[4] != "OK":
			whiteOff()
			blueOn()
		else:
			whiteOff()
			greenOn()

		# turn off the LED if any data is not complete
		if get_location()['nSat'] == 0:
			whiteOff()

		# Wait 0.5 second before gathering knew data!
		time.sleep(0.5)
		i = i + 1

	elif button_status == 0:
		print ("IDLE...")
		whiteOff()
		yellowOn()
		time.sleep(0.5)
		whiteOff()
		time.sleep(0.5)

	elif button_status == 2:
		end = 1

#Shutdown sequence
GPIO.cleanup()
data.close()
whiteOff()
redOn()
print ("Shutdown")
#os.system("sudo shutdown -h now")
whiteOff()