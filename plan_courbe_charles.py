import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Import plotly package
import plotly.graph_objs as go
import numpy as np
import pandas as pd

# Read file of data given by the LAN3
File = pd.read_csv("Data.csv", sep=",")
x = np.array(File['R/G'])
y = np.array(File['B/G'])
z = np.array(File['MSI'])

data = np.c_[x, y, z]

# Create a 3D graph with the values of R/G/B linking to MSI values
trace1 = go.Scatter3d(
    x=data[:, 0],
    y=data[:, 1],
    z=data[:, 2],
    mode='markers',
    marker=dict(size=4, color='red', line=dict(color='black', width=0.5), opacity=0.8)
)

# regular grid covering the domain of the data
mn = np.min(data, axis=0)
mx = np.max(data, axis=0)
X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
XX = X.flatten()
YY = Y.flatten()

# Obtaining the coefficient of the equation giving the MSI values depending on the R/G/B values
A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
print(C)


# evaluate it on a grid
Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)

# plot points and fitted surface using Matplotlib
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
plt.xlabel('R/G')
plt.ylabel('B/G')
ax.set_zlabel('MSI')

axis = dict(
    showbackground=True,  # show axis background
    backgroundcolor="rgb(204, 204, 204)",  # set background color to grey
    gridcolor="rgb(255, 255, 255)",  # set grid line color
    zerolinecolor="rgb(255, 255, 255)",  # set zero grid line color)
)

# plot points and fitted surface using Plotly

trace3 = go.Surface(z=Z, x=X, y=Y, colorscale='RdBu', opacity=0.999)

# Package the trace dictionary into a data object

data_test2 = go.Data([trace1, trace3])

# Make a layout object
layout = go.Layout(
    title='2nd-order (quadratic) surface',  # set plot title
    scene=go.Scene(  # axes are part of a 'scene' in 3d plots
        xaxis=go.XAxis(axis),  # set x-axis style
        yaxis=go.YAxis(axis),  # set y-axis style
        zaxis=go.ZAxis(axis)),  # set z-axis style
)

# Make a figure object
fig = go.Figure(data=data_test2, layout=layout)

plt.show()
plt.legend()
