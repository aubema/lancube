import scipy.linalg
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Import plotly package
import plotly.graph_objs as go
import numpy as np
import pandas as pd

# Read file of data given by the LAN3
File = pd.read_csv("Final_used_Data.csv", sep=",")
x = np.array(File['R/G'])
y = np.array(File['B/G'])
z = np.array(File['MSI'])

data = np.c_[x, y, z]

# regular grid covering the domain of the data
mn = np.min(data[:,:2], axis=0)
mx = np.max(data[:,:2], axis=0)
X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
XX = X.flatten()
YY = Y.flatten()

#define order of the equation to fit. all members must be written by np.prod products
A = np.c_[np.ones(data.shape[0]),                               #order 0: C
        data[:, :2],                                            #order 1: x,y
        np.prod((x,y), axis=0),data[:, :2] ** 2,                #order 2: x*y, x^2, y^2
        np.prod((x,x,y), axis=0),                               #order 3: x^2*y, y^2*x, x^3, y^3
        np.prod((y,y,x), axis=0),
        data[:, :2] ** 3]                                       
                                                


C, _, _, _ = scipy.linalg.lstsq(A, z)
print(C)


# evaluate it on a grid
Z = np.dot(np.c_[np.ones(XX.shape),
                 XX, YY, 
                 XX * YY, XX ** 2, YY ** 2,
                 XX**2*YY, XX*YY**2, XX**3, YY**3], C).reshape(X.shape)

# plot points and fitted surface using Matplotlib
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(x, y, z, c='r', s=50)
plt.xlabel('R/G')
plt.ylabel('B/G')
ax.set_zlabel('MSI')
plt.show()


def Msi_Lan3(x,y,coeff):
    return coeff[0]+coeff[1]*x+coeff[2]*y+coeff[3]*x*y+coeff[4]*x**2+coeff[5]*y**2+\
            coeff[6]*x**2*y+coeff[7]*x*y**2+coeff[8]*x**3+coeff[9]*y**2

def lin_func(x,a,b):
    return a*x+b

msi_lan3=Msi_Lan3(x,y,C)

Clin, _ = scipy.optimize.curve_fit(lin_func, z, msi_lan3)


fig,ax = plt.subplots()
ax.scatter(z, msi_lan3, label='lan3 MSI-3rd order')
ax.plot(z, lin_func(z,Clin[0],Clin[1]), c='k', label='linear fit')
ax.text(0, 0.5, 'm=ax+b \n a={:.2f} \n b={:.2f}'.format(Clin[0],Clin[1]), fontsize=20)
ax.legend(loc='lower right')
plt.show()
