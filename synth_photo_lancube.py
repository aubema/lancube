import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import all files from LSPDD spectra bank folder
fnames = glob('/LSPDD/lampdump/*')

# Load all spectra (MxNxL) files to array and response curves for lan3 (MxN)
spectra = np.array([pd.read_csv(fname).values for fname in fnames])
resps = pd.read_csv('response_spectra_lancube.csv', sep=' ', header='None').values

# Find spectra with keyword (INC, LED, MH, ...)
#incands = spectra[np.argwhere(['INC' in fname for fname in fnames])]

# Plot all spectra with offset to look for outliers
figure()
for i in range(spectra.shape[0]):
	plt.plot(spectra[i,:,1]/spectra[i,:,1].max()+i*0.01,linewidth=0.5)
plt.show()

# Interpolate resps to spectra wavelength values
resps_interp = np.array([np.interp(spectra[0,:,0],resps[:,0],resps[:,i]) for i in [1,2,3,4]])

# Multiply both spectra and responses into single array (Spectra, R, G, B, C)
spectra_resps = np.array([np.multiply(spectra[:,:,1],resps_interp[0]),
						np.multiply(spectra[:,:,1],resps_interp[1]),
						np.multiply(spectra[:,:,1],resps_interp[2]),
						np.multiply(spectra[:,:,1],resps_interp[3])])


# Plot an example normalized spectrum with normalized responses filters applied
figure()
num = 83
plt.plot(spectra[0,:,0],spectra_resps[0,num]/spectra_resps[0,num].max(),color='r', label='R')
plt.plot(spectra[0,:,0],spectra_resps[1,num]/spectra_resps[1,num].max(),color='g', label='G')
plt.plot(spectra[0,:,0],spectra_resps[2,num]/spectra_resps[2,num].max(),color='b', label='B')
plt.plot(spectra[0,:,0],spectra_resps[3,num]/spectra_resps[3,num].max(),color='k', label='C')
plt.legend()
plt.title('Normalized spectrum (#{}) through filter responses'.format(num))
plt.xlabel('$\lambda$(nm)')
plt.ylabel('$I$(a.u.)')






