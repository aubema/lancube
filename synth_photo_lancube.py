import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.interpolate import interp1d


# Import all files from LSPDD spectra bank folder
fnames = glob('LSPDD/lampdump/*')

# Load all spectra (MxNxL) files to array and response curves for lan3 (MxN)
spectra = np.array([pd.read_csv(fname).values for fname in fnames])
resps = pd.read_csv('response_spectra_lancube.csv', sep=' ', header=None).values

# Find spectra with keyword (INC, LED, MH, ...)
#incands = spectra[np.argwhere(['INC' in fname for fname in fnames])]

# Plot all spectra with offset to look for outliers
plt.figure()
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
plt.figure()
num = 83
plt.plot(spectra[0,:,0],spectra_resps[0,num]/spectra_resps[0,num].max(),color='r', label='R')
plt.plot(spectra[0,:,0],spectra_resps[1,num]/spectra_resps[1,num].max(),color='g', label='G')
plt.plot(spectra[0,:,0],spectra_resps[2,num]/spectra_resps[2,num].max(),color='b', label='B')
plt.plot(spectra[0,:,0],spectra_resps[3,num]/spectra_resps[3,num].max(),color='k', label='C')
plt.legend()
plt.title('Normalized spectrum (#{}) through filter responses'.format(num))
plt.xlabel('$\lambda$(nm)')
plt.ylabel('$I$(a.u.)')
plt.show()

# Read indices/sensitivities spectra and reference D65 bulb
msas = pd.read_csv('bio_ref/Msas.csv', sep=',').values
pas = pd.read_csv('bio_ref/Pas.csv', sep=',').values
scotopic = pd.read_csv('bio_ref/Scotopic.csv', sep=',').values
photopic = pd.read_csv('bio_ref/Photopic.csv', sep=',').values
d65 = pd.read_csv('bio_ref/D65.csv', sep=',').values

# Interpolate D65 wavelength data to other indices
msasi = np.interp(msas[:,0], d65[:,0], d65[:,1])

# Integrate interval (wavelength resolution)
dl = msas[1,1]-msas[0,1]

# Normalize D65 (Denominator of the AuRobyfaj equation)
d65_norm = msasi/(dl*np.sum(msasi*photopic[:,1]))


# Function to compute MSI, SLI and IPI indices
def CalcIndices(spectrum):
	
	# Interpolate spectrum wavelength data to other indices
	spectrumi = np.interp(msas[:,0], spectrum[:,0],spectrum[:,1])

	# Normalize spectrum (numerator)
	spectrumi_norm = spectrumi/(dl*np.sum(spectrumi*photopic[:,1]))

	# AuRobyfaj equation
	MSI = np.sum(spectrumi_norm*msas[:,1])/np.sum(d65_norm*msas[:,1])
	SLI = np.sum(spectrumi_norm*scotopic[:,1])/np.sum(d65_norm*scotopic[:,1])
	IPI = np.sum(spectrumi_norm*pas[:,1])/np.sum(d65_norm*pas[:,1])

	return np.array([MSI,SLI,IPI])


# Compute indices
indices = np.array([CalcIndices(spectrum) for spectrum in spectra])  

# Check and correct for outliers (Must verify outliers_list to be sure it's really outliers!)
# Threshold defines max boundary
thresh = 2
outliers_MSI = np.where(indices[:,0]>thresh)[0]
outliers_SLI = np.where(indices[:,1]>thresh)[0]
outliers_IPI = np.where(indices[:,2]>thresh)[0]

# Index of outliers
outliers = np.unique(np.concatenate((outliers_MSI,outliers_SLI,outliers_IPI)))

# List of filenames of outliers
outliers_list = [fnames[outlier] for outlier in outliers]

mask = np.ones(indices[:,0].shape, bool)
mask[outliers] = False
indices_mask = indices[mask]

# Make histogram of values for each index
plt.figure()
plt.hist(indices_mask[:,0],100,label='MSI', color='b', alpha=0.5)
plt.hist(indices_mask[:,1],100,label='SLI', color='k', alpha=0.5)
plt.hist(indices_mask[:,2],100,label='IPI', color='g', alpha=0.5)
plt.xlabel('Index value')
plt.ylabel('Counts')
plt.legend()

# Save data to file
z=np.vstack((np.array(fnames)[mask],indices_mask[:,0],indices_mask[:,1],indices_mask[:,2])).T
np.savetxt('computed_indices.csv', z, header='fname, MSI, SLI, IPI, ', delimiter=',', fmt='%s')