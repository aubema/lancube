# Lancube
![GitHub contributors](https://img.shields.io/github/contributors/aubema/lancube?style=plastic)

LANcube instrument software

<img src="lan3v2.png" width="250">

## Getting started with the Lancube
To install the software system on a physical Lancube please refer to this guide:
https://lx02.cegepsherbrooke.qc.ca/~aubema/index.php/Prof/LAN3v2-technical



## Generate discrete lights inventory from the Lancube
The script ```lancube_inventory.py``` genereate a discreate light source inventory from the data scanned.

The script automatically correct abberrant and imprecise data by looking at the following elements:
  * Disconection from the GPS
  * Sensors with overexpose values
  * Distance between two measures
  * Distance with the roads

For further details on the script execution and the calculations performs, please refer to this paper: xxx


### Support
For any question please refer to this guide: https://lx02.cegepsherbrooke.qc.ca/~aubema/index.php/Prof/Page
