import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Reading data
Street_Spectro = pd.read_csv('/Users/amarfarkouh/PycharmProjects/Map/Indices_Calculator'
                             '/Chambre_Amar/45_blanc_16h30.csv')
Msas = pd.read_csv('/Users/amarfarkouh/PycharmProjects/Map/Indices_Calculator'
                   '/Biological_References/Msas.csv', sep=',')
Pas = pd.read_csv('/Users/amarfarkouh/PycharmProjects/Map/Indices_Calculator'
                  '/Biological_References/Pas.csv', sep=',')
Scotopic = pd.read_csv('/Users/amarfarkouh/PycharmProjects/Map/Indices_Calculator'
                       '/Biological_References/Scotopic.csv', sep=',')
Photopic = pd.read_csv('/Users/amarfarkouh/PycharmProjects/Map/Indices_Calculator'
                       '/Biological_References/Photopic.csv', sep=',')
D65 = pd.read_csv('/Users/amarfarkouh/PycharmProjects/Map/Indices_Calculator'
                  '/Biological_References/D65.csv', sep=',')

# x and y components of the above data
    # Street_spectro
x_Street_spectro = np.array(Street_Spectro['Lambda'])
y_Street_spectro_1 = np.array(Street_Spectro['Intensity'])
y_Street_spectro = y_Street_spectro_1[214:914]


    # Msas
x_Msas = np.array(Msas['wavelength'])
y_Msas = np.array(Msas['relativeIntensity'])

    # Pas
x_Pas = np.array(Pas['wavelength'])
y_Pas = np.array(Pas['relativeIntensity'])

    # Scotopic
x_scotopic = np.array(Scotopic['wavelength'])
y_scotopic = np.array(Scotopic['relativeIntensity'])

    # D65
x_D65 = np.array(D65['wavelength'])
y_D65 = np.array(D65['relativeIntensity'])

# Define Photopic data as a function
x_photopic = np.array(Photopic['wavelength'])
y_photopic = np.array(Photopic['relativeIntensity'])


# Interpolating D65 wavelength data to other indices
F = interp1d(x_D65, y_D65)

y_new = F(x_Msas)

# Normalizing phi
# Phi D65 (Denominator of the AuRobyfaj equation)
Phi_D65 = y_new/((x_Msas[1]-x_Msas[0])*np.sum(y_new*y_photopic))

# Street_spectro normalizing (numerator)
Phi_Street_spectro = y_Street_spectro/((x_Msas[1]-x_Msas[0])*np.sum(y_Street_spectro*y_photopic))

dl = x_Msas[1]-x_Msas[0]

# AuRobyfaj equation
MSI = np.sum(Phi_Street_spectro*y_Msas)/np.sum(Phi_D65*y_Msas)
SLI = np.sum(Phi_Street_spectro*y_scotopic)/np.sum(Phi_D65*y_scotopic)
IPI = np.sum(Phi_Street_spectro*y_Pas)/np.sum(Phi_D65*y_Pas)

print('MSI=', MSI)
print('SLI=', SLI)
print('IPI=', IPI)



