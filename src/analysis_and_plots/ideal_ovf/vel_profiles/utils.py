#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import os
from os.path import join, isfile, basename, splitext
import glob
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

#=============================================================================
def haw_mean(var_masked, e1_2D_masked, e2_2D_masked):
    '''
    haw_mean calculates the Horizontal
    spatial Area Weighted Mean.
    var_masked, e1t_masked and e2t_masked 
    are np.nan masked variable and scale factors 
    2D fields respectively.
    '''

    cell_areas = np.multiply(e1_2D_masked, e2_2D_masked)

    haw_m = np.divide( np.multiply(cell_areas, var_masked).sum(skipna=True),
                       cell_areas.sum(skipna=True) )

    return haw_m

#==============================================================================
def vaw_mean(var_masked, e1_3D_masked, e2_3D_masked, e3_3D_masked):

    #cell_areas  = e1_3D_masked * e2_3D_masked
    cell_vol = e1_3D_masked * e2_3D_masked * e3_3D_masked

    vaw_m = (cell_vol * var_masked).sum(skipna=True) / cell_vol.sum(skipna=True)

    return vaw_m

# ==============================================================================
def compute_masks(ds_domain, merge=False):
    """
    Compute masks from domain_cfg Dataset.
    If merge=True, merge with the input dataset.
    Parameters
    ----------
    ds_domain: xr.Dataset
        domain_cfg datatset
    add: bool
        if True, merge with ds_domain
    Returns
    -------
    ds_mask: xr.Dataset
        dataset with masks
    """

    # Extract variables
    k = ds_domain["z_c"] + 1
    top_level = ds_domain["top_level"]
    bottom_level = ds_domain["bottom_level"]

    # Page 27 NEMO book.
    # I think there's a typo though.
    # It should be:
    #                  | 0 if k < top_level(i, j)
    # tmask(i, j, k) = | 1 if top_level(i, j) ≤ k ≤ bottom_level(i, j)
    #                  | 0 if k > bottom_level(i, j)
    tmask = xr.where(np.logical_or(k < top_level, k > bottom_level), 0, np.nan)
    tmask = xr.where(np.logical_and(bottom_level >= k, top_level <= k), 1, tmask)
    tmask = tmask.rename("tmask")

    tmask = tmask.transpose("z_c","y_c","x_c")

    # Need to shift and replace last row/colum with tmask
    # umask(i, j, k) = tmask(i, j, k) ∗ tmask(i + 1, j, k)
    umask = tmask.rolling(x_c=2).prod().shift(x_c=-1)
    umask = umask.where(umask.notnull(), tmask)
    umask = umask.rename("umask")

    # vmask(i, j, k) = tmask(i, j, k) ∗ tmask(i, j + 1, k)
    vmask = tmask.rolling(y_c=2).prod().shift(y_c=-1)
    vmask = vmask.where(vmask.notnull(), tmask)
    vmask = vmask.rename("vmask")

    # Return
    masks = xr.merge([tmask, umask, vmask])
    if merge:
        return xr.merge([ds_domain, masks])
    else:
        return masks


