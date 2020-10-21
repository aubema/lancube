# Voronoi polygons from lan3 sensors data
# Lancube sensors positions (looking towards front of car):
# 1 top, 2 rear, 3 right, 4 front, 5 left, 6 bottom.

import numpy as np
import pandas as pd
from glob import glob

# Find the filenames of all files in the folder
fnames = glob('Data_Sherbrooke/*.CSV')
fnames.sort()

# Import all files from street measurements
data = np.array([pd.read_csv(fname, header=1) for fname in fnames])

# Parse side sensors (3 and 5)
inds_S3 = np.array([np.where(data['Sensor']=='S3')[0] for data in data])
inds_S5 = np.array([np.where(data['Sensor']=='S5')[0] for data in data])

data_clean = np.array([np.array([data['latitude'].values,
								data['longitude'].values,
								data['R'].values,
								data['G'].values,
								data['B'].values,
								data['clear'].values]).T for data in data])

data_S3 = np.concatenate([data_clean[i][inds_S3[i]] for i in range(len(fnames))])
data_S5 = np.concatenate([data_clean[i][inds_S5[i]] for i in range(len(fnames))])

# Mask for null G values (gives error in divide for indices)
data_S3 = np.delete(data_S3, np.argwhere(data_S3[:,3]==0), axis=0)
data_S5 = np.delete(data_S5, np.argwhere(data_S5[:,3]==0), axis=0)

# Indices coefficients (parameters in order of constant to third order, B/G R/G)
coeff_MSI = [-0.7994, 2.1747, 0.8319, -1.4479, 0.4837, -0.3384, -0.3231, 0.3939, -0.3806, 0.0345]
coeff_SLI = [0.4477, -2.0648, 0.5246, 1.3034, 3.3012, -0.6146, -1.5828, 0.0818, -0.6776, 0.1061]
coeff_IPI = [-2.9130, 12.3930, 1.8631, -4.1413, -15.5166, -0.3217,	3.7306,	-0.1123, 5.9502, 0.0596]

def ComputeIndex(x, y, coeff):
    return coeff[9]*(x**3) + coeff[8]*(y**3) + coeff[7]*(y*(x**2)) + coeff[6]*(x*(y**2)) + coeff[5]*(x**2)+\
           coeff[4]*(y**2) + coeff[3]*(x*y) + coeff[2]*x + coeff[1]*y + coeff[0]

MSI_S3 = ComputeIndex(data_S3[:,2]/data_S3[:,3], data_S3[:,4]/data_S3[:,3], coeff_MSI)
SLI_S3 = ComputeIndex(data_S3[:,2]/data_S3[:,3], data_S3[:,4]/data_S3[:,3], coeff_SLI)
IPI_S3 = ComputeIndex(data_S3[:,2]/data_S3[:,3], data_S3[:,4]/data_S3[:,3], coeff_IPI)

MSI_S5 = ComputeIndex(data_S5[:,2]/data_S5[:,3], data_S5[:,4]/data_S5[:,3], coeff_MSI)
SLI_S5 = ComputeIndex(data_S5[:,2]/data_S5[:,3], data_S5[:,4]/data_S5[:,3], coeff_SLI)
IPI_S5 = ComputeIndex(data_S5[:,2]/data_S5[:,3], data_S5[:,4]/data_S5[:,3], coeff_IPI)

#indices_S3 = np.array([data_S3[:,0],data_S3[:,1], MSI_S3, SLI_S3, IPI_S3]).T
#indices_S5 = np.array([data_S5[:,0],data_S5[:,1], MSI_S5, SLI_S5, IPI_S5]).T

indices_S3 = np.array([data_S3[:,0],data_S3[:,1], MSI_S3]).T
indices_S5 = np.array([data_S5[:,0],data_S5[:,1], MSI_S5]).T

indices_S3_clean_lat = np.delete(indices_S3, np.where(indices_S3[:,0]==0), axis=0)
indices_S3_clean = np.delete(indices_S3_clean_lat, np.where(indices_S3[:,1]==0), axis=0)
indices_S5_clean_lat = np.delete(indices_S5, np.where(indices_S5[:,0]==0), axis=0)
indices_S5_clean = np.delete(indices_S5_clean_lat, np.where(indices_S5[:,1]==0), axis=0)

# Save data to file

np.savetxt('street_measurements_MSI.csv', np.vstack([indices_S3_clean, indices_S5_clean]), delimiter=',', header='lat,lon,MSI', comments='', fmt='%10.5f')
