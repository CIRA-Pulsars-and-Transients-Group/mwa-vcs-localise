#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

# For basic algebra and statistics
import numpy as np
import scipy.stats as st
import scipy.spatial as sp
from scipy.ndimage import label

# Astropy
from astropy.coordinates import SkyCoord
from astropy.table import Table

# For visualization
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import cmasher as cm


def snr_reader(
    path_to_file: str,
) -> tuple[SkyCoord, np.ndarray, np.ndarray, np.ndarray]:
    """Read in the input detection file with the TAB centre coordinates and
    measured detection significance.

    Input files is expected to be in CSV format with headers "ra,dec,snr" at least.
    The coordinates should be in "hms" and "dms" format for "ra" and "dec", respectively.

    Args:
        path_to_file (str): path of the CSV text file

    Returns:
        tuple[SkyCoord, np.ndarray, np.ndarray, np.ndarray]:
            Centre coordinates, detection metrics, relative TAB weights
            and a mask that identifies the highest detection significance TAB.
    """

    obs_snr_table = Table.read(path_to_file, format="csv")
    obs_snr = obs_snr_table["snr"].value
    obs_beam_centers = SkyCoord(
        obs_snr_table["ra"],
        obs_snr_table["dec"],
        frame="icrs",
        unit=("hourangle", "deg"),
    )
    obs_mask = obs_snr < obs_snr.max()
    obs_weights = obs_snr[obs_mask] / obs_snr.max()
    return obs_beam_centers, obs_snr, obs_weights, obs_mask


def covariance_estimation(
    obs_snr: np.ndarray,
    obs_mask: np.ndarray,
    obs_weights: np.ndarray,
    nsim: int = 10000,
    plot_cov: bool = True,
) -> tuple[np.ndarray, Figure]:
    """Estimate the covariance between each TAB pair. This is achieved via
    simulation of the ratios of the pairs of TAB detection statistics. For
    the most part, it only captures the covariance introduced by dividing
    each TAB by a nominal "best detection".

    Args:
        obs_snr (np.ndarray): Observed detection metric (i.e., snr) for each TAB
        obs_mask (np.ndarray): The mask which identifies which TABs are to be compared.
        obs_weights (np.ndarray): The relative weights for each observed detection metric in a TAB.
        nsim (int, optional): How many random multivariate draws to make per pair. Defaults to 10000.
        plot_cov (bool, optional): Whether to plot the covariance matrix. Defaults to True.

    Returns:
        tuple[np.ndarray, Figure]: The covariance matrix itself, and a figure that contains the
            covariance matrix plot (or be blank if `plot_cov` is False)
    """
    simulation_snr = st.multivariate_normal(obs_snr).rvs(nsim)
    simulation_ratio = (
        simulation_snr[:, obs_mask] / simulation_snr.T[obs_snr.argmax()][:, None]
    )
    covariance = np.cov(simulation_ratio, rowvar=False)
    if np.all(np.abs(covariance) < 1e-2):
        print("  Covariances are all < abs(1e-2)")
        print(covariance)
    elif np.all(np.abs(covariance) > 0.15):
        print("  Covariances are all > abs(0.15)")
        print(covariance)
    if np.any(np.abs(covariance) > 0.5):
        print("  WARNING: At least one covariance value is > abs(0.5)")
        print(covariance)

    fig = plt.figure(figsize=(20, 10))
    if plot_cov:
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


def chi2_calc(
    tabp_look: np.ndarray,
    obs_mask: np.ndarray,
    obs_snr: np.ndarray,
    obs_weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Compute the chi-squared statistics based on the least-squares regression formalism.

    The input data are multi-dimensional, and we only wish to compute the statistic across
    the coordinate grid dimensions, thus there is some reshaping at different stages to
    ensure the multiplications/summations corresponding to the normal matrix products occur
    in the correct axes.

    Args:
        tabp_look (np.ndarray): An array of TAB power patterns (2D arrays).
        obs_mask (np.ndarray): The mask which identifies which TABs are of interest.
        obs_snr (np.ndarray): Observed detection metric (i.e., snr) for each TAB.
        obs_weights (np.ndarray): The relative weights for each observed detection metric in a TAB.
        cov (np.ndarray): The covariance matrix between TAB pairs.

    Returns:
        np.ndarray: A 2D chi-square map which can be processed to identify statistical peaks
            that correspond to high localisation relative probabilities.
    """
    P_array = tabp_look[obs_mask, ...] / tabp_look[obs_snr.argmax(), ...]
    R_array = obs_weights[:, None, None] - P_array.squeeze()
    cov_inv = np.linalg.inv(cov)
    n_obs = len(obs_snr)
    reshaped_R = np.reshape(R_array, (n_obs - 1, -1))
    C_dot_R = np.reshape(np.dot(cov_inv, reshaped_R), R_array.shape)
    chi2 = np.sum(R_array * C_dot_R, axis=0)

    return chi2


def estimate_errors_from_islands(
    pmap: np.ndarray,
    grid_ra: np.ndarray,
    grid_dec: np.ndarray,
    ra_idx: int,
    dec_idx: int,
    clvl: float,
) -> tuple[float, tuple[float, float] | None, int]:
    """Calculate the symmetrical conservative error based on the probability islands
    and their extent relative to the peak localisation.

    :param pmap: The localisation probability map.
    :type pmap: np.ndarray
    :param grid_ra: The 2-D mesh grid in R.A. that defines the probability map coordinate.
    :type grid_ra: np.ndarray
    :param grid_dec: The 2-D mesh grid in Dec. that defines the probability map coordinate.
    :type grid_dec: np.ndarray
    :param ra_idx: The R.A. grid index corresponding to the peak probability.
    :type ra_idx: int
    :param dec_idx: The Dec. grid index corresponding to the peak probability.
    :type dec_idx: int
    :param clvl: A contour level (in the same units as the probability map) that defines the uncertainty region.
    :type clvl: float

    :return (max_dist, max_dist_pt, num_islands): A tuple containing -
                The maximum distance from the peak localisation pixel to the provided contour level (i.e., the symmetric uncertainty),
                The pixel coordinates corresponding to the maximum distance, and
                The number of probability islands found in the image (typically 1)
    rtype: tuple[float, tuple[float, float] | None, int]
    """
    # Using those contour levels, estimate the maximum distance from the peak to
    # the corresponding contour and take this as the uncertainty. We collect the
    # islands of probability into labelled groups and only use the island which
    # contains the peak probability to calculate the uncertainties.
    peak_ra, peak_dec = (grid_ra[ra_idx, dec_idx], grid_dec[ra_idx, dec_idx])
    contour_mask = pmap >= clvl
    labeled_prob_map, num_islands = label(contour_mask)
    peak_island = labeled_prob_map[ra_idx, dec_idx]
    same_island_pts = np.where(labeled_prob_map == peak_island)

    max_dist = 0.0
    max_dist_pt = None
    for pt in zip(*same_island_pts):
        pt_ra = grid_ra[pt[0], pt[1]]
        pt_dec = grid_dec[pt[0], pt[1]]
        dist = sp.distance.euclidean([peak_ra, peak_dec], [pt_ra, pt_dec])
        if dist > max_dist:
            max_dist = dist
            max_dist_pt = pt

    return max_dist, max_dist_pt, num_islands


def get2Dcdf(s: float) -> float:
    """Compute the Gaussian CDF value for a give sigma value

    :param s: The desired "sigma" quantity used to evaluate the CDF value.
    :type s: float

    :return cdf: The Gaussian CDF value corresponding to the input "sigma" level.
    :rtype: float
    """
    return 1 - np.exp(-0.5 * s**2)


def mahal_error(prob: np.ndarray, sigma: float = 1) -> float | None:
    """Calculate the Mahalanobis radius to provide an error on the localisation,
    under the assumption that the underlying probability density is ~Gaussian.

    :param prob: The 2D probability density map.
    :type prob: np.ndarray
    :param sigma:  The equivalent Gaussian sigma desired to measure. Defaults to 1.
    :type sigma: float
    :return prob_sigma_level: The probability density value associated with the input sigma level.
    If a sensible value cannot be found, the function return None.
    :rtype: float | None
    """
    prob_flat_sorted = np.sort(prob, axis=None)
    prob_flat_sorted_index = np.argsort(prob, axis=None)
    prob_flat_sorted_cumsum = np.cumsum(prob.flatten()[prob_flat_sorted_index])

    # Compute the survival function value
    sf = 1 - get2Dcdf(sigma)

    prob_sigma_level = None
    if len(np.nonzero(prob_flat_sorted_cumsum > sf)[0]) != 0:
        # Index where cumulative sum goes above the corresponding survival function
        index_sigma_above = np.nonzero(prob_flat_sorted_cumsum > sf)[0]
        prob_sigma_level = prob_flat_sorted[index_sigma_above[0]]
    else:
        print("Unable to find a sensible error level.")

    return prob_sigma_level


def localise_and_plot(
    tab0: np.ndarray,
    chi2: np.ndarray,
    grid_ra: np.ndarray,
    grid_dec: np.ndarray,
    obs_beam_centers: SkyCoord,
    obs_beam_snrs: np.ndarray,
    obs_mask: np.ndarray,
    truth_coords: SkyCoord | None = None,
    window: str | None = None,
    show_bestfit_loc: bool = True,
    locfig_lims: str | list | None = None,
) -> Figure:
    """Generate the localisation maps, identify peaks and uncertainties.
    Plot the results and report the best position identified with a
    corresponding uncertainty.

    Args:
        tab0 (np.ndarray): The TAB pattern corresponding to the best initial detection statistic.
        chi2 (np.ndarray): A 2D chi-square map encoding the localisation probabilities.
        grid_ra (np.ndarray): The 2-D mesh grid in R.A. that defines the probability map coordinate.
        grid_dec (np.ndarray): The 2-D mesh grid in Dec. that defines the probability map coordinate.
        obs_beam_centers (SkyCoord): The TAB centre coordinates.
        obs_beam_snrs (np.ndarray): Observed detection metric (i.e., snr) for each TAB.
        obs_mask (np.ndarray): The mask which identifies which TABs are of interest.
        truth_coords (SkyCoord | None, optional): Coordinates of the true source position (for comparison).
            Defaults to None.
        window (str | None, optional): The kind of smoothing approach to apply to the localisation statistic.
            Defaults to None.
        show_bestfit_loc (bool, optional): Whether to show the best localisation cross-hair. Defaults to True.
        locfig_lims (str | list | None, optional): The x- and y-limits to plot in the figure.
            Also accepts the string "zoom" which automatically scales axes and includes a inset figure.
            Defaults to None.

    Returns:
        Figure: The figure containing the localisation plot.
    """

    map_extent = [grid_ra.min(), grid_ra.max(), grid_dec.min(), grid_dec.max()]
    aspect = "auto"
    origin = "lower"
    cmap = cm.sapphire_r
    ctr_ls = [":", "--", "-"]  # outer to inner, in order
    ctr_colors = ["k", "k", "magenta"]

    if window == "gaus" or window == "gaussian":
        scale = 3
        print(
            f"Placing Gaussian window at central TAB position, with variance ~ {scale} * max. TAB separation."
        )
        ctr_coord = np.squeeze(obs_beam_centers[~obs_mask])
        dists = [
            c.to("deg").value for c in ctr_coord.separation(obs_beam_centers[obs_mask])
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

    # Regularise, if required, then compute probabilities
    chi2 = wt * chi2
    lnL = -0.5 * chi2
    prob = np.exp(lnL) / np.exp(lnL).sum()
    # Mask probabilities less than 1-in-10^9
    # This helps avoid plotting issues and problems
    # when summing/using the map to compute other statistics
    # since the VAST majority of values are tiny
    prob[prob < 1e-9] = 0

    # Coordinates associated with minimum chi2
    best_ra_index, best_dec_index = np.unravel_index(np.argmax(prob), prob.shape)
    best_ra, best_dec = (
        grid_ra[best_ra_index, best_dec_index],
        grid_dec[best_ra_index, best_dec_index],
    )
    best_coord = SkyCoord(best_ra, best_dec, unit="deg")
    best_coord_hms = best_coord.to_string("hmsdms", sep=":", precision=2)
    best_coord_deg = best_coord.to_string("decimal", precision=6)

    # Compute the contour levels via the Mahalanobis radius at various
    # equivalent "sigma" levels, under the assumption of a Gaussian distribution
    sigma_levels = [5, 3, 1]
    print(f"Significance intervals set at: {sigma_levels}-sigma")
    contour_levels = np.array([mahal_error(prob, s) for s in sigma_levels])
    sym_err, _, nislands = estimate_errors_from_islands(
        prob, grid_ra, grid_dec, best_ra_index, best_dec_index, contour_levels.min()
    )

    print(f"best position estimate = {best_coord_hms}")
    print(f"                       = {best_coord_deg} deg")
    for isig, sig in enumerate(sigma_levels):
        sig_err, _, nislands = estimate_errors_from_islands(
            prob, grid_ra, grid_dec, best_ra_index, best_dec_index, contour_levels[isig]
        )
        print(f"  {sig}-sigma sym. pos. err. = {sig_err*60:g} arcmin")

    # Prepare the figure and place artist elements
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax1 = fig.add_subplot(1, 1, 1)

    # localisation map
    ax1.imshow(
        prob,
        aspect=aspect,
        extent=map_extent,
        cmap=cmap,
        origin=origin,
        vmin=contour_levels.min(),
    )

    # contours for specific levels of chi2
    ax1_ctr = ax1.contour(
        prob,
        levels=contour_levels,
        linestyles=ctr_ls,
        colors=ctr_colors,
        extent=map_extent,
        origin=origin,
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
    for j, sobs in enumerate(obs_beam_snrs):
        ax1.annotate(
            f"{sobs:g}",
            xy=(obs_beam_centers.ra.deg[j], obs_beam_centers.dec.deg[j]),
            ha="right",
            va="bottom",
            textcoords="offset points",
            xytext=(-4, 0),
        )

    # If set, now zoom on specified region
    if locfig_lims == "zoom":
        # add a zoomed version of the localisation island
        excess = 2 * sym_err
        x1, x2 = best_ra - excess, best_ra + excess
        y1, y2 = best_dec - excess, best_dec + excess
        ax1_img_inset = ax1.inset_axes(
            [0.65, 0.65, 0.34, 0.34], xlim=(x1, x2), ylim=(y1, y2)
        )
        ax1_img_inset.set_aspect(ax1.get_aspect())

        ax1_img_inset.imshow(
            prob,
            aspect=aspect,
            extent=map_extent,
            cmap=cmap,
            interpolation="none",
            origin=origin,
            vmin=contour_levels.min(),
        )
        ax1_img_inset.contour(
            prob,
            levels=contour_levels,
            linestyles=ctr_ls,
            colors=ctr_colors,
            extent=map_extent,
            origin=origin,
        )
        ax1_img_inset.tick_params(axis="both", labelrotation=45, pad=-2)
        ax1.indicate_inset_zoom(ax1_img_inset, edgecolor="black")
        pad = (
            np.array(
                [
                    obs_beam_centers[~obs_mask].separation(o).deg
                    for o in obs_beam_centers[obs_mask]
                ]
            )
            .flatten()
            .max()
        )

        # Adjust the limits, and bias the x-range to have more space on the right
        # so that the inset box is less likely to overalp elements.
        ax1.set_xlim(
            min(obs_beam_centers.ra.deg) - pad,
            max(obs_beam_centers.ra.deg) + 1.5 * pad,
        )
        ax1.set_ylim(
            min(obs_beam_centers.dec.deg) - pad,
            max(obs_beam_centers.dec.deg) + pad,
        )

        ax1_img_inset.xaxis.set_major_locator(mtick.MaxNLocator(5, prune="both"))
        ax1_img_inset.yaxis.set_major_locator(mtick.MaxNLocator(5, prune="both"))
        ax1_img_inset.grid()

        # Best-fit coordinate crosshair
        if show_bestfit_loc:
            ax1_img_inset.errorbar(
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
            ax1.set_title(
                f"Best-fit localisation = ({best_ra:g}, {best_dec:g}) $\pm$ {sym_err:g} deg",
            )

        # Truth Coordinates for comparison
        if truth_coords is not None:
            ax1_img_inset.plot(
                truth_coords.ra.deg,
                truth_coords.dec.deg,
                "or",
                markersize=5,
                mew=1,
                mfc="none",
                label="Truth",
            )

        # Collect and fix legend handles and labels
        ctr_h = ax1_ctr.legend_elements()[0]
        ctr_l = [f"${s}\sigma$" for s in sigma_levels]
        bpt_h, bpt_l = ax1.get_legend_handles_labels()
        ins_h, ins_l = ax1_img_inset.get_legend_handles_labels()
        ax1.legend(
            handles=ctr_h + bpt_h + ins_h,
            labels=ctr_l + bpt_l + ins_l,
            fontsize=12,
            ncols=2,
            loc="lower right",
        )

    elif locfig_lims is not None:
        # This is mostly for debugging as the user would need to explicitly c
        # call chi2_plot here rather than use the commandline.
        ax1.set_xlim(locfig_lims[0], locfig_lims[1])
        ax1.set_ylim(locfig_lims[2], locfig_lims[3])

        # Best-fit coordinate crosshair
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
        if truth_coords is not None:
            ax1.plot(
                truth_coords.ra.deg,
                truth_coords.dec.deg,
                "or",
                markersize=5,
                mew=1,
                mfc="none",
                label="Truth",
            )

        # Collect and fix legend handles and labels
        ctr_h = ax1_ctr.legend_elements()[0]
        ctr_l = [f"${s}\sigma$" for s in sigma_levels]
        bpt_h, bpt_l = ax1.get_legend_handles_labels()
        ax1.legend(
            handles=ctr_h + bpt_h,
            labels=ctr_l + bpt_l,
            fontsize=12,
        )
    else:
        # More debugging, with less information of plot.
        # Truth Coordinates for comparison
        if truth_coords is not None:
            ax1.plot(
                truth_coords.ra.deg,
                truth_coords.dec.deg,
                "or",
                markersize=5,
                mew=1,
                mfc="none",
                label="Truth",
            )
        # Collect and fix legend handles and labels
        ctr_h = ax1_ctr.legend_elements()[0]
        ctr_l = [f"${s}\sigma$" for s in sigma_levels]
        bpt_h, bpt_l = ax1.get_legend_handles_labels()
        ax1.legend(
            handles=ctr_h + bpt_h,
            labels=ctr_l + bpt_l,
            fontsize=12,
        )

    ax1.set_xlabel("Right Ascension (deg)", fontsize=14, ha="center")
    ax1.set_ylabel("Declination (deg)", fontsize=14, ha="center")
    ax1.minorticks_on()
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.tick_params(axis="both", which="both", direction="out", right=True, top=True)

    return fig


def localise(
    detfile: str,
    tabp_look: np.ndarray,
    grid_ra: np.ndarray,
    grid_dec: np.ndarray,
    cov_nsim: int = 10000,
    plot_cov: bool = True,
    truth_coords: SkyCoord | None = None,
    window: str | None = None,
) -> tuple[Figure, Figure]:
    """Execute the localisation procedure.

    Args:
        detfile (str): The file path containing ra, dec and detection significance information.
        tabp_look (np.ndarray): The array of TAB patters for each pointing in `detfile`
        grid_ra (np.ndarray): The 2-D mesh grid in R.A. that defines the probability map coordinate.
        grid_dec (np.ndarray): The 2-D mesh grid in Dec. that defines the probability map coordinate.
        cov_nsim (int, optional): How many random multivariate draws to make per TAB when estimating
            the covariance. Defaults to 10000.
        plot_cov (bool, optional): Whether to plot the covariance matrix. Defaults to True.
        truth_coords (SkyCoord | None, optional): Coordinates of the true source position (for comparison).
            Defaults to None.
        window (str | None, optional): The kind of smoothing approach to apply to the localisation statistic.
            Defaults to None.

    Returns:
        tuple[Figure, Figure]: Two Figure objects containing the localisation map and covariance matrix.
    """

    obs_beam_centers, obs_snr, obs_weights, obs_mask = snr_reader(detfile)
    covariance, cov_fig = covariance_estimation(
        obs_snr, obs_mask, obs_weights, nsim=cov_nsim, plot_cov=plot_cov
    )
    chi2 = chi2_calc(tabp_look, obs_mask, obs_snr, obs_weights, covariance)
    localization_fig = localise_and_plot(
        tabp_look[obs_snr.argmax(), ...].squeeze(),
        chi2,
        grid_ra,
        grid_dec,
        obs_beam_centers,
        obs_snr,
        obs_mask,
        truth_coords=truth_coords,
        window=window,
        show_bestfit_loc=True,
        locfig_lims="zoom",
    )
    return localization_fig, cov_fig
