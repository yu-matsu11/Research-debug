# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:46:30 2025

@author: wangfan
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interp2(X, Y, V, Xq, Yq, method='linear', fill_value=np.nan):
    """
    
    Parameters
    ----------
    X, Y : 1D or 2D arrays
        Coordinates of the grid points (same as meshgrid output or coordinate vectors).
    V : 2D array
        Values at the grid points.
    Xq, Yq : 2D arrays
        Query points.
    method : str
        Interpolation method: 'linear' (default), 'nearest', 'cubic','spline'.
    fill_value : float
        Value used for points outside the interpolation domain.
        
    Returns
    -------
    Vq : 2D array
        Interpolated values at query points.
    """

    
    if X.ndim == 2 and Y.ndim == 2: # if X,Y is meshgrid, only vector is used
        x = X[0, :]
        y = Y[:, 0]
    else:
        x = X
        y = Y

    # interpolation function
    interp_func = RegularGridInterpolator((y, x), V,
                                          method=method,
                                          bounds_error=False,
                                          fill_value=fill_value)
    
    # query points
    pts = np.stack([Yq.ravel(), Xq.ravel()], axis=-1)
    Vq = interp_func(pts)
    
    return Vq.reshape(Xq.shape)
