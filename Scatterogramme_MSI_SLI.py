import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cfit

# reading data
#Data_LAN3 = pd.read_csv('Scan_Sherbrooke/#2RockFo_2020-05-12.csv')
Data_LAN3 = pd.read_csv('Data.csv')


# r,g,b values
R = Data_LAN3['R']
G = Data_LAN3['G']
B = Data_LAN3['B']

MSI_Spectro = Data_LAN3['MSI']

# Functions to convert r,g,b values to indices (MSI, SLI)
# SLI parameters
coeff1 = [0.92676108, 0.03435651, 0.30409609, 1.96169658, 0.12299434, 0.00232827]

# MSI parameters
coeff2 = [0.0683509,0.558028,0.370345,-1.21431, -0.61563, 0.17249, -0.063395, 1.01142,0.894559, -0.544077]



def MSI(x,y, coeff):
    return coeff[0]*(x**3) + coeff[1]*(y**3) + coeff2[2]*(y*(x**2)) + coeff2[3]*(x*(y**2)) + coeff2[4]*(x**2)+\
           coeff[5]*(y**2) + coeff[6]*(x*y) + coeff2[7]*x + coeff2[8]*y + coeff2[9]


M = MSI(R/G, B/G, coeff2)

plt.scatter(MSI_Spectro, M-MSI_Spectro)
plt.xlabel('MSI_Spectro')
plt.ylabel('MSI_LAN3-MSI_spectro')

#print(coeff2)


#print('coeffs: ', coeffs[0])

plt.show()

# np.savetxt('MSI_SLI.csv', M)
