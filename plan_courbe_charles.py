import scipy.linalg
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
import pandas as pd

##Read file of data given by the LAN3
#File = pd.read_csv("computed_indices.csv", sep=",")

File = pd.read_csv("computed_indices.csv", sep=",")
# x = np.array(File['R/G'])
# y = np.array(File['B/G'])
R = np.array(File['R'])
G = np.array(File['G'])
B = np.array(File['B'])
Clear = np.array(File['C'])
MSI = np.array(File['MSI'])
SLI = np.array(File['SLI'])
IPI = np.array(File['IPI'])

##filter values with minimum threshold (low G values give errors on MSI because of uncertainty)
threshold = 0
Cleartind=np.argwhere(Clear>threshold)
Rtind=np.argwhere(R>threshold)
Gtind=np.argwhere(G>threshold)
Btind=np.argwhere(B>threshold)
Rtind=np.argwhere(R<Rtind.max()*0.5)
Btind=np.argwhere(B<Btind.max()*0.5)
rts=np.unique(np.concatenate((Rtind,Gtind,Btind), axis=None))

Cleart=Clear[rts]
Rt=R[rts]
Gt=G[rts]
Bt=B[rts]
MSIt=MSI[rts]
SLIt=SLI[rts]
IPIt=IPI[rts]


# Choose to compute for MSI, SLI or IPI
data=np.c_[Bt/Gt,Rt/Gt,IPIt]
data=data[data[:,1]<4]
indice='IPI'
# data=np.c_[Bt/Gt,Rt/Gt,SLIt]
# indice='SLI'


##regular grid covering the domain of the data
mn = np.min(data[:,:2], axis=0)
mx = np.max(data[:,:2], axis=0)
X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 30), np.linspace(mn[1], mx[1], 30))
XX = X.flatten()
YY = Y.flatten()


## 2nd order poly fit
##define order of the equation to fit. all terms must be written by np.prod products
A = np.c_[np.ones(data.shape[0]),                                               #order 0: C
        data[:, :2],                                                            #order 1: x,y
        np.prod((data[:,0],data[:,1]), axis=0),data[:, :2] ** 2]                #order 2: x*y, x^2, y^2            

C, res, _, _ = np.linalg.lstsq(A, data[:,2])
print('Coeffs:', C)



##evaluate it on a grid
Z = np.dot(np.c_[np.ones(XX.shape),
                 XX, YY, 
                 XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)



# ## 3rd order poly fit
# ##define order of the equation to fit. all terms must be written by np.prod products
# A = np.c_[np.ones(data.shape[0]),                                               #order 0: C
#         data[:, :2],                                                            #order 1: x,y
#         np.prod((data[:,0],data[:,1]), axis=0),data[:, :2] ** 2,                #order 2: x*y, x^2, y^2
#         np.prod((data[:,0],data[:,0],data[:,1]), axis=0),                       #order 3: x^2*y, y^2*x, x^3, y^3
#         np.prod((data[:,0],data[:,1],data[:,1]), axis=0),
#         data[:, :2] ** 3]                                       
                                                


# C, res, _, _ = np.linalg.lstsq(A, data[:,2])
# print('Coeffs:', C)


# ##evaluate it on a grid
# Z = np.dot(np.c_[np.ones(XX.shape),
#                  XX, YY, 
#                  XX * YY, XX ** 2, YY ** 2,
#                  XX**2*YY, XX*YY**2, XX**3, YY**3], C).reshape(X.shape)

##plot points and fitted surface using Matplotlib
# fig = plt.figure(figsize=(7, 7))
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
# ax.scatter(data[:258,0],data[:258,1],data[:258,2], c='k', s=20, label='LSPDD', marker='o')
# ax.scatter(data[258:,0],data[258:,1],data[258:,2], c='g', s=20, label='In-situ', marker='^')
# ax.set_zlim(0,1)
# plt.xlabel('B/G')
# plt.ylabel('R/G')
# ax.set_zlabel(indice)
# #ax.legend()
# plt.show()


# Surface calculation for 2nd order
def Msi_Lan3(x,y,coeff):
    return coeff[0]+coeff[1]*x+coeff[2]*y+coeff[3]*x*y+coeff[4]*x**2+coeff[5]*y**2

# Surface calculation for 3rd order
# def Msi_Lan3(x,y,coeff):
#     return coeff[0]+coeff[1]*x+coeff[2]*y+coeff[3]*x*y+coeff[4]*x**2+coeff[5]*y**2+\
#             coeff[6]*x**2*y+coeff[7]*x*y**2+coeff[8]*x**3+coeff[9]*y**3


msi_lan3=Msi_Lan3(data[:,0],data[:,1],C)

fig,ax = plt.subplots(1,1)
ax.scatter(data[:,2][:258], msi_lan3[:258], label='LSPDD', s=5, c='k', marker='o')
ax.scatter(data[:,2][258:], msi_lan3[258:], label='In-situ', s=5, c='g', marker='^')
ax.plot(np.linspace(0,1.3,100), lin_func(np.linspace(0,1.3,100),1,0), c='k', label='linear fit', linewidth=0.2)
#ax[0].text(0.05, 0.85, 'm=ax+b \n a={:.2f} \n b={:.2f}'.format(Clin[0],Clin[1]), fontsize=10)
#ax[0].legend(loc='lower right')
#ax[0].set_title('Lan3 ' + indice + ' (3rd order)')
ax.set_xlim(0,1.3)
ax.set_ylim(0,1.3)
ax.set_xlabel(indice + ' (spectrum)')
ax.set_ylabel(indice + ' (LANcube)')

fig,ax = plt.subplots(1,1)
ax.scatter(data[:,2][:258], msi_lan3[:258]-data[:,2][:258], label='LSPDD', s=5, c='k', marker='o')
ax.scatter(data[:,2][258:], msi_lan3[258:]-data[:,2][258:], label='In-situ', s=5, c='g', marker='^')
ax.plot(np.linspace(0,1.3,data[:,2].shape[0]),np.zeros(data[:,2].shape[0]), linewidth=0.5, linestyle='--')
#ax[1].set_title('Residues')
ax.set_xlim(0,1.3)
ax.set_ylim(-0.2,0.2)
#ax[1].legend(loc='lower right')
ax.set_xlabel(indice + ' (spectrum)')
ax.set_ylabel(indice + ' (LANcube) - ' + indice + ' (spectrum)')
plt.show()


print('Std:', np.std(msi_lan3-data[:,2]))


# Plot projections of 3d surface
fig= plt.figure()
ax= fig.add_subplot(111, projection= '3d')

x=data[:,0]
y=data[:,1]
z=data[:,2]
ax.scatter(x,y,z, 'k')

ax.plot(x, z, 'b.', markersize=1, zdir='y', zs=3.5)
ax.plot(y, z, 'r.', markersize=1, zdir='x', zs=1.25)
ax.plot(x, y, 'g.', markersize=1, zdir='z', zs=0)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)

ax.set_xlim([0, 1.25])
ax.set_ylim([0, 3.5])
ax.set_zlim([0, 1])

ax.set_xlabel('b/g')
ax.set_ylabel('r/g')
ax.set_zlabel('MSI (spectrum)')

plt.show()
