#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

# For basic algebra and stats
import numpy as np
import scipy.stats as st
import scipy.spatial as sp

# Astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

# For visualization
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
import cmasher as cm


def __beam_reader(data_directory):
    """
    TO BE DEPRECATED - ONLY KEP FOR TESTING.
    Function to read tabp and grid from files on disk
    This function is to be deprecated as we integrate stats.py wit the rest of the package.
    """
    import os

    tabp_file = os.path.join(data_directory, "tabp_look.npy")
    grid_file = os.path.join(data_directory, "grid.npz")

    tabp_look = np.load(tabp_file)
    print("Selecing the first frequency and ignoring all other frequencies")
    tabp_look = tabp_look[:, 0, :, :]
    grid_ra = np.load(grid_file)["arr_0"]
    grid_dec = np.load(grid_file)["arr_1"]
    # product = np.sum(tabp_look.mean(axis=1), axis=0)
    print("tabp dimensions:", tabp_look.shape)
    # print('product dimensions:', product.shape)
    print("RA/Dec grid dimensions:", grid_ra.shape, grid_dec.shape)

    return tabp_look, grid_ra, grid_dec


def snr_reader(path_to_file):
    """
    Function to read SNR per "look". Input is path to file.
    File should be a CSV, at least containing columns labeled as 'ra', 'dec', 'snr'.
    Coordinate columns should be in hms for ra and dms for dec.

    """
    obs_snr_table = Table.read(path_to_file, format="csv")
    obs_snr = obs_snr_table["snr"].value
    obs_beam_centers = SkyCoord(
        obs_snr_table["ra"],
        obs_snr_table["dec"],
        frame="icrs",
        unit=(u.hourangle, u.deg),
    )
    obs_mask = obs_snr < obs_snr.max()
    obs_weights = obs_snr[obs_mask] / obs_snr.max()
    return obs_beam_centers, obs_snr, obs_weights, obs_mask


def beam_plot(beam_cen_coords, tabp, grid_ra, grid_dec, label, contours=True):
    """
    Making a plot of the beam and contours for looks

    """

    tabp_sum = np.sum(tabp, axis=0)

    map_extent = [grid_ra.min(), grid_ra.max(), grid_dec.min(), grid_dec.max()]

    aspect = "equal"

    cmap = cm.get_sub_cmap(cm.cosmic, 0.1, 0.9)
    cmap.set_bad("red")
    contour_cmap = cm.get_sub_cmap(cm.cosmic_r, 0.1, 0.9)
    cmapnorm_sum = colors.Normalize(vmin=1e-5, vmax=0.1, clip=True)
    cmapnorm_indiv = colors.Normalize(vmin=1e-5, vmax=0.05, clip=True)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1_img = ax1.imshow(
        tabp_sum, aspect=aspect, extent=map_extent, cmap=cmap, norm=cmapnorm_sum
    )

    ax1.plot(
        beam_cen_coords.ra.deg,
        beam_cen_coords.dec.deg,
        "Dy",
        mec="k",
        ms=5,
        label="Beam centers",
    )

    if contours:
        for ls, look in enumerate(tabp):
            ct = ax1.contour(
                look,
                origin="image",
                extent=map_extent,
                cmap=contour_cmap,
                norm=cmapnorm_indiv,
                linewidths=0.5,
            )

    ax1.legend(fontsize=18, loc=2)
    ax1.set_xlabel("R.A. (ICRS)", fontsize=18, ha="center")
    ax1.set_ylabel("Dec. (ICRS)", fontsize=18, ha="center")
    ax1.minorticks_on()
    ax1.tick_params(axis="both", which="major", labelsize=18)
    ax1.tick_params(axis="both", which="major", length=9)
    ax1.tick_params(axis="both", which="minor", length=4.5)
    ax1.tick_params(axis="both", which="both", direction="out", right=True, top=True)

    cbar = fig.colorbar(
        ax1_img,
        ax=fig.axes,
        shrink=1,
        orientation="horizontal",
        location="top",
        aspect=30,
        pad=0.02,
    )
    cbar.ax.set_title(label, fontsize=18, ha="center")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(direction="in", length=5, bottom=True, top=True)
    cbar.ax.xaxis.set_tick_params(labelsize=18)
    return fig


def covariance_estimation(obs_snr, obs_mask, obs_weights, nsim=10000, plot_cov=True):
    simulation_snr = st.multivariate_normal(obs_snr).rvs(nsim)
    simulation_ratio = (
        simulation_snr[:, obs_mask] / simulation_snr.T[obs_snr.argmax()][:, None]
    )
    covariance = np.cov(simulation_ratio, rowvar=False)
    if np.all(np.abs(covariance) < 1e-2):
        print("  Covariances are all < abs(1e-2)")
        print(covariance)
    elif np.all(np.ans(covariance) > 0.15):
        print("  Covariances are all > abs(0.15)")
        print(covariance)
    if np.any(np.abs(covariance) > 0.5):
        print("  WARNING: At least one covariance value is > abs(0.5)")
        print(covariance)

    if plot_cov:
        fig = plt.figure(figsize=(20, 10))
        cmap = cm.get_sub_cmap(cm.guppy, 0.0, 1.0)

        ax1 = fig.add_subplot(1, 1, 1)
        ax1_img = ax1.imshow(covariance, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        i_maxsnr = np.argmax(obs_snr) + 1
        beam_pair_labels = np.array(
            [f"{obs_i+1}/{i_maxsnr}" for obs_i, obs_snr in enumerate(obs_snr)]
        )[obs_mask]
        ax1.set_xticks(ticks=np.arange(0, len(obs_weights)), labels=beam_pair_labels)
        ax1.set_yticks(ticks=np.arange(0, len(obs_weights)), labels=beam_pair_labels)
        ax1.set_xlabel("Beam pair", fontsize=24, ha="center")
        ax1.set_ylabel("Beam pair", fontsize=24, ha="center")
        ax1.set_title(r"$i_{\rm SNRmax}=$ " + f"{i_maxsnr}", fontsize=24, va="bottom")
        ax1.tick_params(axis="both", which="major", labelsize=24)
        ax1.tick_params(axis="both", which="major", length=0)
        ax1.tick_params(
            axis="both", which="both", direction="out", right=True, top=True
        )

        cbar = fig.colorbar(
            ax1_img, ax=fig.axes, orientation="vertical", location="right", pad=0.01
        )
        cbar.ax.set_ylabel("Covariance", fontsize=24, rotation=270, labelpad=20)
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.tick_params(
            which="major", direction="in", length=9, left=True, right=True
        )
        cbar.ax.yaxis.set_tick_params(labelsize=24)

    return covariance, fig


def chi2_calc(tabp_look, obs_mask, obs_snr, obs_weights, cov):
    P_array = tabp_look[obs_mask, ...] / tabp_look[obs_snr.argmax(), ...]
    R_array = obs_weights[:, None, None] - P_array.squeeze()
    cov_inv = np.linalg.inv(cov)
    n_obs = len(obs_snr)
    reshaped_R = np.reshape(R_array, (n_obs - 1, -1))
    C_dot_R = np.reshape(np.dot(cov_inv, reshaped_R), R_array.shape)
    chi2 = np.sum(R_array * C_dot_R, axis=0)

    return chi2


def estimate_errors_from_contours(ctr_set):
    max_distance = 0
    for ic, collection in enumerate(ctr_set.collections):
        for path in collection.get_paths():
            # Get the vertices of the contour path
            vertices = path.vertices
            # Compute pairwise distances between vertices
            distances = sp.distance.cdist(vertices, vertices)
            # Get the maximum distance in this set of vertices
            max_distance_in_path = np.max(distances)
            # Update the overall maximum distance if needed
            max_distance = max(max_distance, max_distance_in_path)

    # return half the distance (as a symmetrical error)
    return max_distance / 2


def CDF(s):
    # Cumulative distribution function for 2D Gaussian
    return  1 - np.exp(-0.5 * s ** 2)


def mahal_error(prob, sigma=1):
    likelihood_flat_sorted = np.sort(prob, axis=None)
    likelihood_flat_sorted_index = np.argsort(prob, axis=None)
    likelihood_flat_sorted_cumsum = np.cumsum(prob.flatten()[likelihood_flat_sorted_index])

    if len(np.nonzero(likelihood_flat_sorted_cumsum > (1 - CDF(sigma)))[0]) != 0:
        # Index where cum sum goes above sigma percentage
        index_sigma_above = np.nonzero(likelihood_flat_sorted_cumsum > 
                        (1 - CDF(sigma)))[0]

        # Minimum likelihood included in error
        likelihood_index_sigma_sorted = likelihood_flat_sorted_index[index_sigma_above]

        likelihood_sigma_level = likelihood_flat_sorted[index_sigma_above[0]]

        likelihood_index_sigma_original = np.unravel_index(likelihood_index_sigma_sorted, 
                                                            prob.shape)

        """
        y_lower_index, y_higher_index = np.sort(likelihood_index_sigma_original[0])[[np.s_[0], 
                                                np.s_[-1]]]
        x_lower_index, x_higher_index = np.sort(likelihood_index_sigma_original[1])[[np.s_[0], 
                                                np.s_[-1]]]
        max_likehood_index = np.unravel_index(np.argmax(prob), prob.shape)

        x_lower = x_lower_index, max_likehood_index[0]
        x_higher = x_higher_index, max_likehood_index[0]
        y_lower = max_likehood_index[1], y_lower_index
        y_higher = max_likehood_index[1], y_higher_index
        """
        return likelihood_sigma_level#, [x_lower, x_higher, y_lower, y_higher]

    else:
        print('Could not find error')
        exit()


def chi2_plot(
    tab0,
    chi2,
    grid_ra,
    grid_dec,
    obs_beam_centers,
    obs_mask,
    contour_levels=None,
    truth_coords=None,
    window=None,
    show_bestfit_loc=True,
):
    map_extent = [grid_ra.min(), grid_ra.max(), grid_dec.min(), grid_dec.max()]

    aspect = "auto"
    origin = "lower"
    # cmap = cm.get_sub_cmap(cm.sapphire_r, 0.1, 0.9)
    cmap = cm.sapphire_r
    ctr_cmap = cm.get_sub_cmap(cm.ember, 0.4, 0.9)
    # ctr_cmap = cm.ember
    # cmapnorm = colors.LogNorm()

    if window == "gaus":
        scale = 3
        print(
            f"Placing Gaussian window at central TAB position, with variance ~ {scale} * max. TAB separation."
        )
        ctr_coord = np.squeeze(obs_beam_centers[~obs_mask])
        dists = [
            c.to(u.deg).value for c in ctr_coord.separation(obs_beam_centers[obs_mask])
        ]
        max_dist = max(dists)
        mu = np.array([ctr_coord.ra.deg, ctr_coord.dec.deg])
        var = np.array(
            [
                [max_dist, 0],
                [0, max_dist],
            ]
        )
        kern = st.multivariate_normal(mean=mu, cov=scale * var)
        wt = 1 / kern.pdf(np.dstack((grid_ra, grid_dec)))
    elif window == "tab":
        wt = 1 / tab0
    else:
        wt = 1.0

    chi2 = wt * chi2
    lnL = -0.5 * chi2
    prob = np.exp(lnL) / np.exp(lnL).sum()
    sig_intervals = np.array([68.27, 95.45, 99.73])  # 1, 2, 3-sigma
    print(f"Significance intervals set at: {sig_intervals}")
    if contour_levels == None:
        #contour_levels = np.percentile(prob[prob > 1e-6], sig_intervals)
        contour_levels = np.array([mahal_error(prob[prob > 1e-6], s) for s in [3,2,1] ] )
        print(contour_levels)

    # fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax1 = fig.add_subplot(1, 1, 1)

    # localisation map
    ax1_img = ax1.imshow(
        prob,
        aspect=aspect,
        extent=map_extent,
        cmap=cmap,
        # norm=cmapnorm,
        origin=origin,
        vmin=0.5 * contour_levels.min(),
    )
    # contours for specific levels of chi2
    ax1_ctr = ax1.contour(
        prob,
        levels=contour_levels,
        extent=map_extent,
        origin=origin,
        cmap=ctr_cmap,
    )
    # weighting window
    # if window == "gaus":
    #     fmt = {}
    #     lvls = np.array([0.1, 0.5, 0.9])
    #     strs = lvls.astype(str)
    #     ax1_ctr_win = ax1.contour(
    #         1 / wt,
    #         levels=lvls * (1 / wt).max(),
    #         extent=map_extent,
    #         origin=origin,
    #         colors="w",
    #         linestyles="dotted",
    #     )
    #     for l, s in zip(ax1_ctr_win.levels, strs):
    #         fmt[l] = s
    #     ax1.clabel(ax1_ctr_win, ax1_ctr_win.levels, fmt=fmt, inline=True, fontsize=8)

    # Coordinates associated with minimum chi2
    # best_ra_index, best_dec_index = np.unravel_index(np.argmin(lnL), lnL.shape)
    best_ra_index, best_dec_index = np.unravel_index(np.argmax(prob), prob.shape)
    best_ra, best_dec = (
        grid_ra[best_ra_index, best_dec_index],
        grid_dec[best_ra_index, best_dec_index],
    )

    # Get the errors (taken from the contours).
    # We search all vertices of each contour set and find the maximum distance.
    # If the contours are set at 1,2,3-sigma levels, then the largest distances
    # corresponds roughly to the 3-sigma uncertainty.
    sym_err = estimate_errors_from_contours(ax1_ctr)
    print(f"best R.A. = {best_ra:g} deg, best Dec. = {best_dec:g} deg")
    print(f"sym. pos. err. = {sym_err*60:g} arcmin")

    if show_bestfit_loc:
        ax1.errorbar(
            best_ra,
            best_dec,
            yerr=sym_err,
            xerr=sym_err,
            marker="none",
            color="k",
            markersize=1,
            mew=1,
            label="Best fit localisation",
        )

    # Truth Coordinates for comparison
    if truth_coords != None:
        ax1.plot(
            truth_coords.ra.deg,
            truth_coords.dec.deg,
            "or",
            markersize=3,
            mew=1,
            mfc="none",
            label="Truth",
        )

    # Beams
    ax1.plot(
        obs_beam_centers.ra.deg[obs_mask],
        obs_beam_centers.dec.deg[obs_mask],
        "Dy",
        mec="k",
        ms=5,
        label="Beam centres",
    )
    ax1.plot(
        obs_beam_centers.ra.deg[~obs_mask],
        obs_beam_centers.dec.deg[~obs_mask],
        "Dy",
        mec="r",
        ms=5,
        label="Beam centre with max. S/N",
    )

    ax1.set_xlim(72.9, 73.1)
    ax1.set_ylim(-34.4, -34.2)
    ax1.legend(fontsize=10, loc=2)
    # ax1.set_title(
    #     # f"Localisation: R.A. = {best_ra:g}$\pm${err_ra:g} deg, Dec. = {best_dec:g}$\pm${err_dec:g} deg",
    #     f"Localisation: R.A. = {best_ra:g}$\pm${sym_err:g} deg, Dec. = {best_dec:g}$\pm${sym_err:g} deg",
    #     fontsize=12,
    #     pad=10,
    # )
    ax1.set_xlabel("Right Ascension (deg)", fontsize=14, ha="center")
    ax1.set_ylabel("Declination (deg)", fontsize=14, ha="center")
    ax1.minorticks_on()
    ax1.tick_params(axis="both", which="major", labelsize=12)
    # ax1.tick_params(axis="both", which="major", length=9)
    # ax1.tick_params(axis="both", which="minor", length=4.5)
    ax1.tick_params(axis="both", which="both", direction="out", right=True, top=True)

    cbar = fig.colorbar(
        ax1_img,
        ax=fig.axes,
        # shrink=0.73,
        orientation="vertical",
        location="right",
        # aspect=30,
        pad=0.01,
        extend="min",
    )
    cbar.add_lines(ax1_ctr)
    # cbar.ax.set_title(r"$\chi^2$", fontsize=12, ha="center")
    # cbar.ax.set_title(r"$\ln \mathcal{L}$", fontsize=12, ha="center")
    cbar.ax.set_title(r"$Pr$", fontsize=12, ha="center")
    # cbar.ax.xaxis.set_ticks_position("top")
    # cbar.ax.tick_params(which="major", direction="in", length=9, bottom=True, top=True)
    # cbar.ax.tick_params(which="minor", direction="in", length=5, bottom=True, top=True)
    cbar.ax.yaxis.set_tick_params(labelsize=11)
    return fig


def seekat(
    detfile,
    tabp_look,
    grid_ra,
    grid_dec,
    cov_nsim=10000,
    plot_cov=True,
    truth_coords=None,
):
    obs_beam_centers, obs_snr, obs_weights, obs_mask = snr_reader(detfile)
    covariance, cov_fig = covariance_estimation(
        obs_snr, obs_mask, obs_weights, nsim=cov_nsim, plot_cov=plot_cov
    )
    chi2 = chi2_calc(tabp_look, obs_mask, obs_snr, obs_weights, covariance)
    localization_fig = chi2_plot(
        tabp_look[obs_snr.argmax(), ...].squeeze(),
        chi2,
        grid_ra,
        grid_dec,
        obs_beam_centers,
        obs_mask,
        contour_levels=None,
        truth_coords=truth_coords,
        window="tab",
        show_bestfit_loc=True,
    )
    return localization_fig, cov_fig
