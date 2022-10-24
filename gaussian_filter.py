# Use a Gaussian filter to determine the height category of lights fixtures
# Author : Julien-Pierre Houle
# May 2022

import numpy as np
import pandas as pd
from pylab import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None



def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)


def gaussian_filter(df_invent, h):

    H = df_invent['H'].values
    mask_H = H > h
    idx_mask = np.argwhere(H > h).flatten()
    H = H[mask_H]

    plt.figure()
    y, x, _ = plt.hist(H, bins=int(H.max()//2), alpha=.3)
    plt.xlabel('H (m)')
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)


    # Trying to find Gaussians
    p0 = (11, 5, y[5])
    params, cov = curve_fit(gauss, x, y, p0)
    plt.plot(x,gauss(x,*params),color='red',lw=3,label='original fit')
    plt.legend()
    plt.savefig('/home/jhoule42/Documents/Lancube/Figures/gauss_filter')
    plt.show()


    lights_class = np.array([3.5, 11, 30]) # mettre comme params


    # # Looking for keeping only physical properties
    # idx_pos_sigma = np.argwhere(sigma > 0).flatten()
    # idx_close_mean = np.argwhere( (avg > lights_class*0.5) & (avg < lights_class*1.5) ).flatten()
    # idx_pos_ampl = np.argwhere(ampl > 0).flatten()
    # mask_class = np.intersect1d(idx_pos_sigma, idx_close_mean)
    # mask_class = np.intersect1d(mask_class, idx_pos_ampl)
    # avg  = avg[mask_class]
    # sigma = sigma[mask_class]
    # ampl = ampl[mask_class]


    H[H < lights_class[1]*0.5] = lights_class[0]

    H[(lights_class[1]*0.5 <= H) & (H <= lights_class[2]*0.85)] = lights_class[1]

    H[H > lights_class[2]*0.85] = lights_class[2]

    # Update height in the inventory
    df_invent['H'].loc[idx_mask] = H
    
    # Rename latitude and longitude
    

    return df_invent
