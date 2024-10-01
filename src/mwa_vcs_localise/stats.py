#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

# For basic algebra and stats
import numpy as np
import scipy.stats as st

# Astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

# For visualization
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
import cmasher as cms


def beam_reader(data_directory):
    """
    TO BE DEPRECATED - ONLY KEP FOR TESTING.
    Function to read tabp and grid from files on disk
    This function is to be deprecated as we integrate stats.py wit the rest of the package.
    When deprecation is confirmed, statements in lines 173 and 287 in make_tab_pattern should be obsolete.
    """
    tabp_file = os.path.join(data_directory,'tabp_look.npy')
    grid_file = os.path.join(data_directory,'grid.npz')
    
    tabp_look = np.load(tabp_file)
    print("Selecing the first frequency and ignoring all other frequencies")
    tabp_look = tabp_look[:,0,:,:]
    grid_ra = np.load(grid_file)['arr_0']
    grid_dec = np.load(grid_file)['arr_1']
    #product = np.sum(tabp_look.mean(axis=1), axis=0)
    print('tabp dimensions:', tabp_look.shape)
    #print('product dimensions:', product.shape)
    print('RA/Dec grid dimensions:', grid_ra.shape, grid_dec.shape)
    
    return tabp_look, grid_ra, grid_dec


def snr_reader(path_to_file):
    """
    Function to read SNR per "look". Input is path to file.
    File should be a CSV, at least containing columns labeled as 'ra', 'dec', 'snr'.
    Coordinate columns should be in hms for ra and dms for dec.

    """
    obs_snr_table = Table.read(path_to_file, format='csv')
    obs_snr = obs_snr_table['snr'].value
    obs_beam_centers = SkyCoord(obs_snr_table['ra'],obs_snr_table['dec'],frame='icrs',unit=(u.hourangle, u.deg))
    obs_mask = obs_snr < obs_snr.max()
    obs_weights = obs_snr[obs_mask]/obs_snr.max()
    return obs_beam_centers, obs_snr, obs_weights, obs_mask


def beam_plot(beam_cen_coords, tabp, grid_ra, grid_dec, label, contours=True):
    """
    Making a plot of the beam and contours for looks

    """


    tabp_sum = np.sum(tabp, axis=0)

    map_extent = [grid_ra.min(),
                  grid_ra.max(),
                  grid_dec.min(),
                  grid_dec.max()]
    
    aspect = 'equal'    

    cmap = cms.get_sub_cmap(cms.cosmic,0.1,0.9)
    cmap.set_bad('red')
    contour_cmap = cms.get_sub_cmap(cms.cosmic_r,0.1,0.9)
    cmapnorm_sum = colors.Normalize(vmin=1e-5, vmax=0.1, clip=True)
    cmapnorm_indiv = colors.Normalize(vmin=1e-5, vmax=0.05, clip=True)
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(1,1,1)
    ax1_img = ax1.imshow(tabp_sum,
                         aspect=aspect, extent=map_extent, cmap=cmap, norm=cmapnorm_sum)

    ax1.plot(beam_cen_coords.ra.deg,beam_cen_coords.dec.deg,'Dy',mec='k', ms=5, label='Beam centers')

    if contours:
        for ls, look in enumerate(tabp):
            ct = ax1.contour(look, origin='image', extent=map_extent, cmap=contour_cmap, norm=cmapnorm_indiv, linewidths=0.5)
    
    ax1.legend(fontsize=18, loc=2)
    ax1.set_xlabel('R.A. (ICRS)',fontsize=18,ha='center')
    ax1.set_ylabel('Dec. (ICRS)',fontsize=18,ha='center')
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.tick_params(axis='both', which='major', length=9)
    ax1.tick_params(axis='both', which='minor', length=4.5)
    ax1.tick_params(axis='both', which='both', direction='out', right=True, top=True)

    cbar = fig.colorbar(ax1_img, ax=fig.axes, shrink=1, orientation='horizontal',location='top', aspect=30, pad=0.02)
    cbar.ax.set_title(label,fontsize=18,ha='center')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(direction='in',length=5,bottom=True,top=True)
    cbar.ax.xaxis.set_tick_params(labelsize=18)
    return fig


def covariance_estimation(obs_snr, obs_mask, obs_weights, nsim=1000, plot_cov=True):
    simulation_snr = st.multivariate_normal(obs_snr).rvs(nsim)
    simulation_ratio = simulation_snr[:,obs_mask]/simulation_snr.T[obs_snr.argmax()][:,None]
    covariance = np.cov(simulation_ratio,rowvar=False)

    if plot_cov:
        fig = plt.figure(figsize=(20,10))
        cmap = cms.get_sub_cmap(cms.guppy,0.0,1.0)

        ax1 = fig.add_subplot(1,1,1)
        ax1_img = ax1.imshow(covariance, cmap=cmap, vmin=-1,vmax=1)
        i_maxsnr = np.argmax(obs_snr)+1
        beam_pair_labels = np.array([f'{obs_i+1}/{i_maxsnr}' for obs_i, obs_snr in enumerate(obs_snr)])[obs_mask]
        ax1.set_xticks(ticks=np.arange(0,len(obs_weights)),labels=beam_pair_labels)
        ax1.set_yticks(ticks=np.arange(0,len(obs_weights)),labels=beam_pair_labels)
        ax1.set_xlabel('Beam pair',fontsize=24,ha='center')
        ax1.set_ylabel('Beam pair',fontsize=24,ha='center')
        ax1.set_title(r'$i_\textrm{SNRmax}=$ '+f'{i_maxsnr}',fontsize=24,va='bottom')
        ax1.tick_params(axis='both', which='major', labelsize=24)
        ax1.tick_params(axis='both', which='major', length=0)
        ax1.tick_params(axis='both', which='both', direction='out', right=True, top=True)

        cbar = fig.colorbar(ax1_img, ax=fig.axes, orientation='vertical',location='right', pad=0.01)
        cbar.ax.set_ylabel('Covariance',fontsize=24,rotation=270,labelpad=20)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.tick_params(which='major',direction='in',length=9,left=True,right=True)
        cbar.ax.yaxis.set_tick_params(labelsize=24)
    
    return covariance, fig


def chi2_calc(tabp_look, obs_mask, obs_snr, obs_weights, cov):
    P_array = tabp_look[obs_mask,...]/tabp_look[obs_snr.argmax(),...]
    R_array = obs_weights[:,None,None]-P_array
    cov_inv = np.linalg.inv(cov)
    n_obs = len(obs_snr)
    reshaped_R = np.reshape(R_array, (n_obs-1, -1))
    C_dot_R = np.reshape(np.dot(cov_inv, reshaped_R), R_array.shape)
    chi2 = np.sum(R_array * C_dot_R, axis=0)

    return chi2


def chi2_plot(chi2, grid_ra, grid_dec, obs_beam_centers, obs_mask, contour_levels=[20,40,60,100], truth_coords=None):
    map_extent = [grid_ra.min(),
                  grid_ra.max(),
                  grid_dec.min(),
                  grid_dec.max()]

    aspect = 'equal'

    cmap = cms.get_sub_cmap(cms.cosmic_r,0.1,0.9)
    cmapnorm = colors.LogNorm(vmin=1e1, vmax=1e5, clip=False)
    contour_cmap = cms.get_sub_cmap(cms.ember,0.4,0.9)
    
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(1,1,1)

    # chi2 map
    ax1_img = ax1.imshow(chi2, aspect=aspect, extent=map_extent, cmap=cmap, norm=cmapnorm)
    # contours for specific levels of chi2
    ax1_ctr = ax1.contour(chi2, levels=contour_levels, extent=map_extent, origin='image', cmap=contour_cmap)

    # Coordinates associated with minimum chi2 
    best_ra_index, best_dec_index = np.unravel_index(np.argmin(chi2), chi2.shape)
    best_ra, best_dec = grid_ra[best_ra_index, best_dec_index], grid_dec[best_ra_index, best_dec_index]
    ax1.plot(best_ra, best_dec,'xk', markersize=10,mew=3,label='Best fit')

    # Truth Coordinates for comparison
    if truth_coords:
        ax1.plot(truth_coords.ra.deg, truth_coords.dec.deg, '+w', markersize=10,mew=3,label='Truth')

    # Beams
    #ax1.plot(look_ra[OBS_MASK], look_dec[OBS_MASK], 'Dy',mec='k', ms=10, label='Look beam centers')
    #ax1.plot(look_ra[~OBS_MASK], look_dec[~OBS_MASK], 'Dy',mec='r', ms=10, label='Look beam center with max SNR')
    ax1.plot(obs_beam_centers.ra.deg[obs_mask], obs_beam_centers.dec.deg[obs_mask], 'Dy', mec='k', ms=10, label='Beam centers')
    ax1.plot(obs_beam_centers.ra.deg[~obs_mask], obs_beam_centers.dec.deg[~obs_mask], 'Dy', mec='r', ms=10, label='Beam center with max SNR')
    
    ax1.legend(fontsize=18, loc=2)
    ax1.set_xlabel('R.A. (ICRS)',fontsize=18,ha='center')
    ax1.set_ylabel('Dec. (ICRS)',fontsize=18,ha='center')
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.tick_params(axis='both', which='major', length=9)
    ax1.tick_params(axis='both', which='minor', length=4.5)
    ax1.tick_params(axis='both', which='both', direction='out', right=True, top=True)

    cbar = fig.colorbar(ax1_img, ax=fig.axes, shrink=0.73, orientation='horizontal',location='top', aspect=30, pad=0.02)
    cbar.add_lines(ax1_ctr)
    cbar.ax.set_title(r'$\chi^2$',fontsize=18,ha='center')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(which='major',direction='in',length=9,bottom=True,top=True)
    cbar.ax.tick_params(which='minor',direction='in',length=5,bottom=True,top=True)
    cbar.ax.xaxis.set_tick_params(labelsize=18)
    return fig


def main():
    