# Lancube
![GitHub contributors](https://img.shields.io/github/contributors/aubema/lancube?style=plastic)

LANcube instrument software

<img src="lan3v2.png" width="250">

## Getting started with the Lancube
* Build your own lancube: [guide](http://57.134.100.114:2080/wiki/index.php/Prof/LAN3v2-users-manual).
* Install the software system on a physical Lancube please refer to this [guide](http://57.134.100.114:2080/wiki/index.php/Prof/LAN3v2-technical).




## Generate discrete lights inventory
You simply need to execute the script ```make_inventory.py``` to generates a discrete light source inventory from the data scanned. Make sure to **modify the paramters** in the file before execution.


By calling ```cleaning_data.py```, the software will automatically correct aberrant and imprecise data by looking at the following elements:
  * Disconnection from the GPS
  * Sensors with overexpose values
  * Distance between two measures
  * Distance with the roads

### Inventory output
During execution the file ```lan3_inventory.csv``` will be created in your execution folder.
It provides the following lamps characteristic:
   * GPS coordinates: (lat, lon)
   * H: Light heigth
   * h: Lancube heigth
   * D: Distance with prime
   * d: Distance from light
   * Lights technologie
   * RGB / Clear


For further details on the script execution and the calculations performs, please refer to this paper: xxx


### Support
For any question please refer to this guide: http://57.134.100.114:2080/wiki/index.php/Prof/LAN3v2-users-manual
