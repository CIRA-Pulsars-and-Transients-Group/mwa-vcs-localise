########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

# For basic algebra and statistics
import cmasher as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.spatial as sp
import scipy.stats as st

# Astropy
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS

# For visualisation
from matplotlib.figure import Figure
from scipy.ndimage import label


def snr_reader(
    path_to_file: str,
) -> tuple[SkyCoord, np.ndarray, np.ndarray, np.ndarray]:
    """Read beam-centre coordinates and S/N values from a detection CSV file.

    The CSV is expected to include at least ``ra``, ``dec``, and ``snr``
    columns, where RA/Dec values are in hourangle/degree string format.

    :param path_to_file: Path to the CSV file.
    :type path_to_file: str
    :returns: Beam-centre coordinates, S/N values, relative TAB weights, and
        a mask selecting non-maximum-S/N entries.
    :rtype: tuple[SkyCoord, np.ndarray, np.ndarray, np.ndarray]
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
    """Estimate covariance between TAB ratios via Monte Carlo simulation.

    :param obs_snr: Observed S/N value for each TAB.
    :type obs_snr: np.ndarray
    :param obs_mask: Mask selecting TABs to compare against the max-S/N TAB.
    :type obs_mask: np.ndarray
    :param obs_weights: Relative weights derived from observed S/N values.
    :type obs_weights: np.ndarray
    :param nsim: Number of random draws used to estimate covariance.
    :type nsim: int
    :param plot_cov: If True, render a covariance heatmap.
    :type plot_cov: bool
    :returns: Estimated covariance matrix and its diagnostic figure.
    :rtype: tuple[np.ndarray, Figure]
    """
    simulation_snr = st.multivariate_normal(obs_snr).rvs(nsim)
    simulation_ratio = (
        simulation_snr[:, obs_mask]
        / simulation_snr.T[obs_snr.argmax()][:, None]
    )
    covariance = np.cov(simulation_ratio, rowvar=False)
    if np.all(np.abs(covariance) < 0.2):
        print("Covariances are all < abs(0.2)")
    elif np.all(np.abs(covariance) < 0.5):
        print("Covariances are all > abs(0.5)")
    else:
        print("WARNING: At least one covariance value is > abs(0.5)")
        print(covariance)

    fig = plt.figure(figsize=(20, 10))
    if plot_cov:
        cmap = cm.get_sub_cmap(cm.guppy, 0.0, 1.0)
        vlim = max([np.abs(covariance.min()), covariance.max()])
        ax1 = fig.add_subplot(1, 1, 1)
        ax1_img = ax1.imshow(
            covariance,
            cmap=cmap,
            vmin=-vlim,
            vmax=vlim,
            aspect="auto",
        )
        i_maxsnr = np.argmax(obs_snr) + 1
        beam_pair_labels = np.array(
            [
                f"{obs_i + 1}/{i_maxsnr}"
                for obs_i, obs_snr in enumerate(obs_snr)
            ]
        )[obs_mask]
        ax1.set_xticks(
            ticks=np.arange(0, len(obs_weights)), labels=beam_pair_labels
        )
        ax1.set_yticks(
            ticks=np.arange(0, len(obs_weights)), labels=beam_pair_labels
        )
        ax1.set_xlabel("Beam pair", fontsize=24, ha="center")
        ax1.set_ylabel("Beam pair", fontsize=24, ha="center")
        ax1.set_title(
            r"$i_{\rm SNRmax}=$ " + f"{i_maxsnr}", fontsize=24, va="bottom"
        )
        ax1.tick_params(axis="both", which="major", labelsize=24)
        ax1.tick_params(axis="both", which="major", length=0)
        ax1.tick_params(
            axis="both", which="both", direction="out", right=True, top=True
        )

        cbar = fig.colorbar(
            ax1_img,
            ax=fig.axes,
            orientation="vertical",
            location="right",
            pad=0.01,
        )
        cbar.ax.set_ylabel(
            "Covariance", fontsize=24, rotation=270, labelpad=20
        )
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
    """Compute the localisation chi-squared surface on the sky grid.

    :param tabp_look: TAB power patterns.
    :type tabp_look: np.ndarray
    :param obs_mask: Mask selecting TABs used in the fit.
    :type obs_mask: np.ndarray
    :param obs_snr: Observed S/N value for each TAB.
    :type obs_snr: np.ndarray
    :param obs_weights: Relative weights for observed TAB measurements.
    :type obs_weights: np.ndarray
    :param cov: Covariance matrix between TAB-ratio measurements.
    :type cov: np.ndarray
    :returns: Two-dimensional chi-squared map over the sampled sky grid.
    :rtype: np.ndarray
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
    """Estimate symmetric localisation uncertainty from contour islands.

    :param pmap: Localisation probability map.
    :type pmap: np.ndarray
    :param grid_ra: RA coordinate grid corresponding to ``pmap``.
    :type grid_ra: np.ndarray
    :param grid_dec: Dec coordinate grid corresponding to ``pmap``.
    :type grid_dec: np.ndarray
    :param ra_idx: RA index of the peak-probability pixel.
    :type ra_idx: int
    :param dec_idx: Dec index of the peak-probability pixel.
    :type dec_idx: int
    :param clvl: Contour level used to define the uncertainty region.
    :type clvl: float
    :returns: Maximum peak-to-contour distance, its pixel coordinate, and the
        number of connected islands.
    :rtype: tuple[float, tuple[float, float] | None, int]
    """

    # Using the provided contour level, estimate the maximum distance from the peak to
    # the corresponding contour and take this as the uncertainty. We collect the
    # islands of probability into labelled groups and only use the island which
    # contains the peak probability to calculate the uncertainties.
    peak_ra, peak_dec = (grid_ra[ra_idx, dec_idx], grid_dec[ra_idx, dec_idx])
    contour_mask = pmap >= clvl
    labelled_prob_map, num_islands = label(contour_mask)
    peak_island = labelled_prob_map[ra_idx, dec_idx]
    same_island_pts = np.where(labelled_prob_map == peak_island)

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
    """Compute the 2D Gaussian CDF value at a given sigma level.

    :param s: Sigma level.
    :type s: float
    :returns: CDF value corresponding to ``s``.
    :rtype: float
    """
    return 1 - np.exp(-0.5 * s**2)


def mahal_error(prob: np.ndarray, sigma: float = 1) -> float | None:
    """Map an equivalent Gaussian sigma level to a probability contour value.

    :param prob: Two-dimensional probability density map.
    :type prob: np.ndarray
    :param sigma: Gaussian-equivalent sigma level.
    :type sigma: float
    :returns: Probability density contour value for ``sigma``, or ``None`` if
        no sensible contour is found.
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
    wcs: WCS | None,
    obs_beam_centers: SkyCoord,
    obs_beam_snrs: np.ndarray,
    obs_mask: np.ndarray,
    truth_coords: SkyCoord | None = None,
    window: str | None = None,
    show_bestfit_loc: bool = True,
    zoom: bool = True,
) -> Figure:
    """Render localisation probability maps, contours, and annotations.

    :param tab0: TAB pattern for the maximum-S/N detection.
    :type tab0: np.ndarray
    :param chi2: Chi-squared map used to derive localisation probability.
    :type chi2: np.ndarray
    :param grid_ra: RA coordinate grid.
    :type grid_ra: np.ndarray
    :param grid_dec: Dec coordinate grid.
    :type grid_dec: np.ndarray
    :param wcs: WCS projection for plotting.
    :type wcs: WCS | None
    :param obs_beam_centers: TAB centre coordinates.
    :type obs_beam_centers: SkyCoord
    :param obs_beam_snrs: Observed S/N values for each TAB.
    :type obs_beam_snrs: np.ndarray
    :param obs_mask: Mask selecting non-maximum-S/N TABs.
    :type obs_mask: np.ndarray
    :param truth_coords: Optional reference coordinate for comparison.
    :type truth_coords: SkyCoord | None
    :param window: Optional regularization scheme (for example ``tab`` or
        ``gaussian``).
    :type window: str | None
    :param show_bestfit_loc: If True, show best-fit crosshair in inset.
    :type show_bestfit_loc: bool
    :param zoom: If True, include an inset around the best-fit region.
    :type zoom: bool
    :returns: Matplotlib figure containing the localisation visualisation.
    :rtype: Figure
    """

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
            c.to("deg").value
            for c in ctr_coord.separation(obs_beam_centers[obs_mask])
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
    best_ra_index, best_dec_index = np.unravel_index(
        np.argmax(prob), prob.shape
    )
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
    sym_err, _, _ = estimate_errors_from_islands(
        prob,
        grid_ra,
        grid_dec,
        best_ra_index,
        best_dec_index,
        contour_levels.min(),
    )

    print(f"best position estimate = {best_coord_hms}")
    print(f"                       = {best_coord_deg} deg")
    for isig, sig in enumerate(sigma_levels):
        sig_err, _, _nislands = estimate_errors_from_islands(
            prob,
            grid_ra,
            grid_dec,
            best_ra_index,
            best_dec_index,
            contour_levels[isig],
        )
        print(f"  {sig}-sigma sym. pos. err. = {sig_err * 60:g} arcmin")

    # Prepare the figure and place artist elements
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax1 = fig.add_subplot(1, 1, 1, projection=wcs)

    # Localisation map
    ax1.imshow(
        prob,
        aspect=aspect,
        cmap=cmap,
        origin=origin,
        vmin=contour_levels.min(),
    )

    # Contours for specific levels of chi2
    ax1_ctr = ax1.contour(
        prob,
        levels=contour_levels,
        linestyles=ctr_ls,
        colors=ctr_colors,
        origin=origin,
    )

    # Beams and S/N measurements
    ax1.plot(
        obs_beam_centers.ra.deg[obs_mask],
        obs_beam_centers.dec.deg[obs_mask],
        "Dy",
        mec="k",
        ms=5,
        label="Beam centres",
        transform=ax1.get_transform("world"),
    )
    ax1.plot(
        obs_beam_centers.ra.deg[~obs_mask],
        obs_beam_centers.dec.deg[~obs_mask],
        "Dy",
        mec="r",
        ms=5,
        label="Beam centre with max. S/N",
        transform=ax1.get_transform("world"),
    )
    for j, sobs in enumerate(obs_beam_snrs):
        ax1.annotate(
            f"{sobs:.1f}",
            xy=(obs_beam_centers.ra.deg[j], obs_beam_centers.dec.deg[j]),
            xytext=(4, 0),
            xycoords=ax1.get_transform("world"),
            ha="left",
            va="bottom",
            textcoords="offset points",
        )

    if zoom:
        # Padding around the centre of the island in the inset
        inset_pad = 1.5 * sym_err
        ix1, ix2 = best_ra + inset_pad, best_ra - inset_pad
        iy1, iy2 = best_dec - inset_pad, best_dec + inset_pad
        ibb_pix = wcs.world_to_pixel(
            SkyCoord([ix1, ix2], [iy1, iy2], frame="icrs", unit="deg")
        )

        # Position the inset in the top-right corner of the figure
        ax1_img_inset = ax1.inset_axes(
            [0.675, 0.675, 0.31, 0.31], projection=wcs
        )
        ax1_img_inset.set_aspect(ax1.get_aspect())

        # Localisation map (inset)
        ax1_img_inset.imshow(
            prob,
            aspect=aspect,
            cmap=cmap,
            interpolation="none",
            origin=origin,
            vmin=contour_levels.min(),
        )

        # Contours for specific levels of chi2 (inset)
        ax1_img_inset.contour(
            prob,
            levels=contour_levels,
            linestyles=ctr_ls,
            colors=ctr_colors,
            origin=origin,
        )

        # Set inset limits based on bounding box padding (bb_pix) calculated above
        ax1_img_inset.set_xlim(ibb_pix[0])
        ax1_img_inset.set_ylim(ibb_pix[1])

        # Draw the inset bounding box on the parent axis
        ax1.indicate_inset_zoom(ax1_img_inset, edgecolor="black")

        # Set some axis customisations
        ax1_img_inset.xaxis.set_major_locator(
            mtick.MaxNLocator(5, prune="both")
        )
        ax1_img_inset.yaxis.set_major_locator(
            mtick.MaxNLocator(5, prune="both")
        )
        ax1_img_inset.grid(ls=":")
        ax1_img_inset.tick_params(axis="both", direction="out")
        ax1_img_inset.set_xlabel(" ")
        ax1_img_inset.set_ylabel(" ")

        # Add a best-fit coordinate crosshair in the inset, and include best-fit position in figure title
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
                transform=ax1_img_inset.get_transform("world"),
            )
            ax1.set_title(
                f"""Best-fit localisation = {best_coord_hms}\nUncertainty ({sigma_levels[0]}$\\sigma$, no iono.) = $\\pm$ {sym_err * 60:g} arcmin""",
            )

        # Add a truth coordinate for comparison to the inset
        if truth_coords is not None:
            ax1_img_inset.plot(
                truth_coords.ra.deg,
                truth_coords.dec.deg,
                "or",
                markersize=5,
                mew=1,
                mfc="none",
                transform=ax1_img_inset.get_transform("world"),
            )

    # Add a truth coordinate for comparison to the main axis
    if truth_coords is not None:
        ax1.plot(
            truth_coords.ra.deg,
            truth_coords.dec.deg,
            "or",
            markersize=5,
            mew=1,
            mfc="none",
            label="Truth",
            transform=ax1.get_transform("world"),
        )

        best_true_sep = best_coord.separation(truth_coords)
        print(
            f"Offset of truth from best-fit position: {best_true_sep.to('arcmin'):g}"
        )

    # Collect and fix legend handles and labels
    ctr_h = ax1_ctr.legend_elements()[0]
    ctr_l = [rf"${s}\sigma$" for s in sigma_levels]
    bpt_h, bpt_l = ax1.get_legend_handles_labels()

    all_handles = ctr_h + bpt_h
    all_labels = ctr_l + bpt_l

    if zoom:
        ins_h, ins_l = ax1_img_inset.get_legend_handles_labels()
        all_handles += ins_h
        all_labels += ins_l

    ax1.legend(
        handles=all_handles,
        labels=all_labels,
        fontsize=12,
        ncols=2,
        loc="lower right",
    )

    # Padding around the centre of the island in the parent axis
    # with addition padding on the right and bottom sides so that
    # insets and legends don't overlap data points.
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
    x1, x2 = (
        max(obs_beam_centers.ra.deg) + 0.3 * pad,
        min(obs_beam_centers.ra.deg) - 1.8 * pad,
    )
    y1, y2 = (
        min(obs_beam_centers.dec.deg) - pad,
        max(obs_beam_centers.dec.deg) + 0.3 * pad,
    )

    bb_pix = wcs.world_to_pixel(
        SkyCoord(
            [x1, x2],
            [y1, y2],
            frame="icrs",
            unit="deg",
        )
    )

    # Customise the parent axis limits, labels and grids
    if zoom:
        ax1.set_xlim(bb_pix[0])
        ax1.set_ylim(bb_pix[1])
    ax1.set_xlabel("Right Ascension", fontsize=14, ha="center")
    ax1.set_ylabel("Declination", fontsize=14, ha="center")
    ax1.grid(ls=":")
    ax1.minorticks_on()
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.tick_params(axis="both", which="both", direction="out")

    return fig


def localise(
    detfile: str,
    tabp_look: np.ndarray,
    grid_ra: np.ndarray,
    grid_dec: np.ndarray,
    wcs: WCS | None,
    cov_nsim: int = 10000,
    plot_cov: bool = True,
    truth_coords: SkyCoord | None = None,
    window: str | None = None,
    zoom: bool = True,
) -> tuple[Figure, Figure]:
    """Execute the full localisation workflow from detections and TAB maps.

    :param detfile: CSV file containing at least ``ra``, ``dec``, and ``snr``.
    :type detfile: str
    :param tabp_look: TAB power maps for the sampled look directions.
    :type tabp_look: np.ndarray
    :param grid_ra: RA coordinate grid.
    :type grid_ra: np.ndarray
    :param grid_dec: Dec coordinate grid.
    :type grid_dec: np.ndarray
    :param wcs: WCS projection used by localisation plots.
    :type wcs: WCS | None
    :param cov_nsim: Number of simulations used for covariance estimation.
    :type cov_nsim: int
    :param plot_cov: If True, plot the covariance matrix.
    :type plot_cov: bool
    :param truth_coords: Optional known source coordinate for comparison.
    :type truth_coords: SkyCoord | None
    :param window: Optional regularization scheme.
    :type window: str | None
    :param zoom: If True, include a zoomed inset in localisation plot.
    :type zoom: bool
    :returns: Localization figure and covariance-matrix figure.
    :rtype: tuple[Figure, Figure]
    """

    obs_beam_centers, obs_snr, obs_weights, obs_mask = snr_reader(detfile)
    covariance, cov_fig = covariance_estimation(
        obs_snr, obs_mask, obs_weights, nsim=cov_nsim, plot_cov=plot_cov
    )
    chi2 = chi2_calc(tabp_look, obs_mask, obs_snr, obs_weights, covariance)
    localisation_fig = localise_and_plot(
        tabp_look[obs_snr.argmax(), ...].squeeze(),
        chi2,
        grid_ra,
        grid_dec,
        wcs,
        obs_beam_centers,
        obs_snr,
        obs_mask,
        truth_coords=truth_coords,
        window=window,
        show_bestfit_loc=True,
        zoom=zoom,
    )
    return localisation_fig, cov_fig
