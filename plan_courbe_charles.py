import scipy.linalg
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
import pandas as pd

##Read file of data given by the LAN3
File = pd.read_csv("computed_indices.csv", sep=",")
# x = np.array(File['R/G'])
# y = np.array(File['B/G'])
z = np.array(File['MSI'])
R = np.array(File['R'])
G = np.array(File['G'])
B = np.array(File['B'])
#data = np.c_[x, y, z]

##filter values with minimum threshold (low G values give errors on MSI because of uncertainty)
threshold = 1000
Rtind=np.argwhere(R>threshold)
Gtind=np.argwhere(G>threshold)
Btind=np.argwhere(B>threshold)
Rtind=np.argwhere(R<Rtind.max()*0.5)
Btind=np.argwhere(B<Btind.max()*0.5)
rts=np.unique(np.concatenate((Rtind,Gtind,Btind), axis=None))

Rt=R[rts]
Gt=G[rts]
Bt=B[rts]
MSIt=z[rts]

data=np.c_[Bt/Gt,Rt/Gt,MSIt]

##regular grid covering the domain of the data
mn = np.min(data[:,:2], axis=0)
mx = np.max(data[:,:2], axis=0)
X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 30), np.linspace(mn[1], mx[1], 30))
XX = X.flatten()
YY = Y.flatten()

##define order of the equation to fit. all members must be written by np.prod products
A = np.c_[np.ones(data.shape[0]),                                               #order 0: C
        data[:, :2],                                                            #order 1: x,y
        np.prod((data[:,0],data[:,1]), axis=0),data[:, :2] ** 2,                #order 2: x*y, x^2, y^2
        np.prod((data[:,0],data[:,0],data[:,1]), axis=0),                       #order 3: x^2*y, y^2*x, x^3, y^3
        np.prod((data[:,0],data[:,1],data[:,1]), axis=0),
        data[:, :2] ** 3]                                       
                                                


C, res, _, _ = np.linalg.lstsq(A, MSIt)
print(C)


##evaluate it on a grid
Z = np.dot(np.c_[np.ones(XX.shape),
                 XX, YY, 
                 XX * YY, XX ** 2, YY ** 2,
                 XX**2*YY, XX*YY**2, XX**3, YY**3], C).reshape(X.shape)

##plot points and fitted surface using Matplotlib
fig = plt.figure(figsize=(5, 5))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:176,0],data[:176,1],data[:176,2], c='b', s=20, label='LSPDD')
ax.scatter(data[176:,0],data[176:,1],data[176:,2], c='r', s=20, label='Jo')
ax.set_zlim(0,1)
plt.xlabel('R/G')
plt.ylabel('B/G')
ax.set_zlabel('MSI')
ax.legend()
plt.show()


def Msi_Lan3(x,y,coeff):
    return coeff[0]+coeff[1]*x+coeff[2]*y+coeff[3]*x*y+coeff[4]*x**2+coeff[5]*y**2+\
            coeff[6]*x**2*y+coeff[7]*x*y**2+coeff[8]*x**3+coeff[9]*y**2

def lin_func(x,a,b):
    return a*x+b

msi_lan3=Msi_Lan3(data[:,0],data[:,1],C)

Clin, _ = scipy.optimize.curve_fit(lin_func, MSIt, msi_lan3, p0=[1,0])


fig,ax = plt.subplots(1,2)
ax[0].scatter(MSIt[:176], msi_lan3[:176], label='LSPDD', s=0.75, c='b')
ax[0].scatter(MSIt[176:], msi_lan3[176:], label='Jo', s=0.75, c='r')
ax[0].plot(MSIt, lin_func(MSIt,Clin[0],Clin[1]), c='k', label='linear fit')
ax[0].text(0, 1.2, 'm=ax+b \n a={:.2f} \n b={:.2f} \n threshold={}'.format(Clin[0],Clin[1], threshold), fontsize=10)
ax[0].legend(loc='lower right')
ax[0].set_title('Lan3 MSI-3rd order')
ax[1].scatter(MSIt[:176], msi_lan3[:176]-MSIt[:176], label='LSPDD', s=0.75, c='b')
ax[1].scatter(MSIt[176:], msi_lan3[176:]-MSIt[176:], label='Jo', s=0.75, c='r')
ax[1].set_title('Residues')
#ax[1].plot(MSIt, lin_func(MSIt,Clin[0],Clin[1]), c='k', label='linear fit')
#ax[1].text(0, 0.5, 'm=ax+b \n a={:.2f} \n b={:.2f} \n threshold={}'.format(Clin[0],Clin[1], threshold), fontsize=20)
ax[1].legend(loc='lower right')

plt.show()


