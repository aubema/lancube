import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cfit

# reading data
#Data_LAN3 = pd.read_csv('Scan_Sherbrooke/#2RockFo_2020-05-12.csv')
Data_LAN3 = pd.read_csv('Courbe/plan_courbe_charles/Data.csv')


# r,g,b values
R = Data_LAN3['R']
G = Data_LAN3['G']
B = Data_LAN3['B']
MSI_Spectro = Data_LAN3['MSI']

# Functions to convert r,g,b values to indices (MSI, SLI)
# SLI parameters
coeff1 = [0.92676108, 0.03435651, 0.30409609, 1.96169658, 0.12299434, 0.00232827]

# MSI parameters
coeff2 = [-0.037105, -0.175255, 0.0811561, -1.44443, 1.08942, 2.03727, 1.35882, -0.685629, -1.52301, -0.102361]


def SLI(x,y):
    return -coeff1[0]*(y**2)+coeff1[1]*(x**2)-coeff1[2]*(x*y)+coeff1[3]*y-coeff1[4]*x-coeff1[5]


def MSI(x,y):
    return coeff2[0]*(x**3)+coeff2[1]*(y**3)+coeff2[2]*(y*(x**2))+coeff2[3]*(x*(y**2))+coeff2[4]*(x**2)+\
           coeff2[5]*(y**2) + coeff2[6]*(x*y) + coeff2[7]*x + coeff2[8]*y + coeff2[9]


M = MSI(R/G, B/G)
S = SLI(R/G, B/G)

def linear_func(x,a,b):
    return a*x+b

coeffs = cfit(linear_func, MSI_Spectro, M)


xs=np.linspace(0,1,1001)
fit_ys=linear_func(xs,coeffs[0][0],coeffs[0][1])


plt.scatter(MSI_Spectro, M-MSI_Spectro)
#plt.plot(xs,fit_ys)
#plt.plot(x,x, c='k')
#plt.xlim(0, 1)
#plt.ylim(0, 1)
plt.xlabel('MSI_Spectro')
plt.ylabel('MSI_LAN3 ')

#print(coeff2)


#print('coeffs: ', coeffs[0])

plt.show()

# np.savetxt('MSI_SLI.csv', M)
