import numpy as np
import matplotlib.pyplot as plt
import time
import RPi.GPIO as GPIO
import SDL_DS3231

x = []
print("setting up gpio")
# general GPIO setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# button GPIO septup
GPIO.setup(3, GPIO.OUT)
GPIO.setup(5, GPIO.OUT)
GPIO.setup(7, GPIO.OUT)

GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(19, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

print("setting up variables")

button_status = 0
end = 0
wait = 0
wait_1 = 0

ds3231 = SDL_DS3231.SDL_DS3231(1, 0x68)
today = str(ds3231.read_datetime())
year = today[0:4]
month = today[5:7]
day = today[8:10]

name = "{}-{}-{}.csv".format(year, month, day)

print("virables set")

while end == 0:
    if GPIO.input(21) == 1:
        button_status = 1
    elif GPIO.input(19) == 1:
        button_status = 2
    else:
        button_status = 0

    if button_status == 1:
        if wait == 0:
            print("starting data acquisition, waiting 10 sec...")
            time.sleep(10)
            print("waiting for 100 data...")
         
        wait = 1
        data = np.genfromtxt("/var/www/html/data/" + name, delimiter=",", names=["", "", "", "", "", "z", "b", "", "", "", "", "", "", "", "y", "", "", "", "", "", ""])

        if len(data['z'])/5 > 110:
            if wait_1 == 0:
                print("starting...")

            wait_1 = 1

            for i in range(len(data['z'])):
                x.append((data['z'][i]*60) + data['b'][i])

            x = x[1::5]

            y_s1 = data['y'][1::5]
            y_s2 = data['y'][2::5]
            y_s3 = data['y'][3::5]
            y_s4 = data['y'][4::5]
            y_s5 = data['y'][5::5]

            plt.clf()
            #plt.ylim(0, 2000)
            plt.plot(x[-100:], y_s1[-100:])
            plt.plot(x[-100:], y_s2[-100:])
            plt.plot(x[-100:], y_s3[-100:])
            plt.plot(x[-100:], y_s4[-100:])
            plt.plot(x[-100:], y_s5[-100:])

            plt.savefig('/var/www/html/data/live_graph/graph.jpg')

            time.sleep(0.3)
    elif button_status == 2:
        plt.plot(0, 0)
        plt.savefig('/var/www/html/data/live_graph/graph.jpg')
        print("Shutdown...")
        break
    elif button_status == 0:
        print("waiting...")
        time.sleep(1)
