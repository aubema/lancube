# Plot graphs from the Lancube algorithm
# Author : Julien-Pierre Houle
# May 2022

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PATH_FIGURE = '/home/jhoule42/Documents/Lancube/Figures'
PATH_DATA = '/home/jhoule42/Documents/Lancube/Data'


# def graph_spectra(df1, df35, ED1_all, EV1_all, ED35, EV35):

# Spectra Raw
# Spectra Raw + Peaks + Primes
plt.figure()
plt.plot(df1['distance'], df1['Value'], label='Top sensor')
plt.plot(df3['distance'], df3['Value'], label='Side sensor')

plt.xlim(83170, 83950)
plt.ylim(-0.01, 0.19)
plt.xlabel('Traveled distance (m)')
plt.ylabel('Clear / Gain / Aquisition Time')  # Unités
plt.legend()
plt.savefig(f"{PATH_FIGURE}/RAW")
plt.close()


# Spectra Raw + Peaks
# Spectra Raw + Peaks + Primes
plt.figure()
plt.plot(df1['distance'], df1['Value'], label='Top sensor')
plt.plot(df3['distance'], df3['Value'], label='Side sensor')

plt.plot(ED1, EV1, 'x', c='red', label='Lights detection') # Peaks
plt.plot(ED3, EV3, 'x', c='red') # Peaks

plt.xlim(83170, 83950)
plt.ylim(-0.01, 0.19)
plt.xlabel('Traveled distance (m)')
plt.ylabel('Clear / Gain / Aquisition Time')  # Unités
plt.legend()
plt.savefig(f"{PATH_FIGURE}/RAW_peaks")
plt.close()




# Spectra Raw + Peaks + Primes
plt.figure()
plt.plot(df1['distance'], df1['Value'], label='Top sensor')
plt.plot(df3['distance'], df3['Value'], label='Side sensor')

plt.plot(ED1, EV1, 'x', c='red', label='Lights detection') # Peaks
plt.plot(ED3, EV3, 'x', c='red') # Peaks

plt.plot(ED1p, EV1p, 'o', c='g', label='Close value') # Prime
plt.plot(ED3p, EV3p, 'o', c='g') # Prime

plt.xlim(83170, 83950)
plt.ylim(-0.01, 0.19)
plt.xlabel('Traveled distance (m)')
plt.ylabel('Clear / Gain / Aquisition Time')  # Unités
plt.legend()
plt.savefig(f"{PATH_FIGURE}/RAW_peaks_prime")
plt.close()






    
# # Graph lights tech
# df_lights = pd.read_csv(f'{PATH_DATA}/spectrum_colors.csv')
# plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(df_lights['r/g'].values*0.14, df_lights['b/g'].values, df_lights['i/g'].values)
# ax.set_xlabel('r/g')
# ax.set_ylabel('b/g')
# ax.set_zlabel('i/g')
# # plt.savefig(f'{PATH_FIGURE}/lights_3d')
# plt.show()