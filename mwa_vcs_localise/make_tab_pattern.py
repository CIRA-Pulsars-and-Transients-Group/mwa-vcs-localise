#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
import time as timer
import argparse
import mwalib
import numpy as np
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
from astropy.constants import c as sol
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmasher as cmr
from .utils import (
    MWA_LOCATION,
    form_grid_positions,
    find_max_baseline,
    plot_array_layout,
    makeMaskFromWeightedPatterns,
)
from .array_factor import (
    extractWorkingTilePositions,
    calcGeometricDelays,
    calcArrayFactorPower,
)
from .primary_beam import getPrimaryBeamPower

plt.rcParams.update(
    {
        "font.family": "serif",
    }
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        dest="metafits",
        type=str,
        help="Metafits file of the associated observation.",
    )
    parser.add_argument(
        "-t", dest="time", type=str, help="UTC time of observation (format: ISOT)."
    )
    parser.add_argument(
        "-f",
        dest="freq",
        nargs="+",
        type=float,
        help="Observing frequency in Hz. Maximum of 5.",
    )
    parser.add_argument(
        "-L",
        dest="look",
        type=str,
        help="Look-direction sky position (format: 'hh:mm:ss_dd:mm:ss')",
    )
    parser.add_argument(
        "-P",
        dest="position",
        type=str,
        help="""Sky position to compute array factor given the look-direction
        (format: 'hh:mm:ss_dd:mm:ss').
        You may provide multiple sky positions to sample, separating them by
        a single <space>.
        You can define a box around the look-direction to simulate using the --gridbox option.
        If no argument provided, will sample the sky around position based on
        an estimate of the FWHM. """,
        default=None,
    )
    # parser.add_argument(
    #     "--nlayers",
    #     type=int,
    #     help="""If not argument provided to -P, this option sets how many
    #     circular layers of FWHM-sized cells around the central position to
    #     calculate.""",
    #     default=100,
    # )
    parser.add_argument(
        "--gridbox",
        type=str,
        help="""Coordinates (RA/Dec) defining the box to sample. 
        Format is a single string as follows: 'RA0 Dec0 RA1 Dec1 RAstep Decstep' in h:m:s d:m:s,
        where (RA0, Dec0) is one corner and (RA1, Dec1) is the opposite corner, and '*step' 
        is the grid pixel size in arcseconds.""",
        default=None,
    )
    parser.add_argument(
        "--nopb",
        action="store_true",
        help="Don't include the primary beam attenuation.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the TAB power for each provided point.",
    )

    args = parser.parse_args()
    if len(args.freq) > 5:
        print("Cannot use more than 5 frequencies at a time, please adjust input.")
        exit(1)
    freqs = np.array(args.freq)
    if args.gridbox:
        grid_box = args.gridbox.split(" ")[:-2]
        grid_step = args.gridbox.split(" ")[-2:]

    tt0 = timer.time()
    print("Preparing metadata...")
    # Collect meta information and setup configuration.
    context = mwalib.MetafitsContext(args.metafits)
    max_baseline, _, _ = find_max_baseline(context)
    tile_positions = extractWorkingTilePositions(context)
    fwhm = (1.22 * (sol.value / freqs) / max_baseline) * u.rad
    print(f"... maximum baseline (m): {max_baseline}")
    print(f"... beam fwhm (arcmin): {fwhm.to(u.arcminute).value}")
    time = Time(args.time, format="isot", scale="utc")
    altaz_frame = AltAz(location=MWA_LOCATION, obstime=time)
    print("... plotting array layout")
    plot_array_layout(context)

    # Create the astrometric quantity for the beamformed target direction
    print("Creating look-direction vector...")
    look_ras = []
    look_decs = []
    for p in args.look.split(" "):
        look_ras.append(p.split("_")[0])
        look_decs.append(p.split("_")[1])

    look_positions = SkyCoord(
        look_ras,
        look_decs,
        frame="icrs",
        unit=("hourangle", "deg"),
    )
    print("Converting to AltAz...")
    t0 = timer.time()
    look_positions_altaz = look_positions.transform_to(altaz_frame)
    t1 = timer.time()
    print(f"... took {t1-t0} seconds")

    # In principle, allow the user to provide N inputs separated by spaces, or just
    # ask for M pointings around the source
    target_ras = []
    target_decs = []

    t0 = timer.time()
    print(
        "Creating sky position vectors from highest frequency and first look-direction..."
    )
    if not args.position and args.gridbox:
        box = SkyCoord(
            [grid_box[0], grid_box[2]],
            [grid_box[1], grid_box[3]],
            frame="icrs",
            unit=("hourangle", "deg"),
        )
        grid_step_ra = grid_step[0] * u.arcsec
        grid_step_dec = grid_step[1] * u.arcsec
        spherical_offset = box[0].spherical_offsets_to(box[1])
        n_ra = int(np.abs(spherical_offset[0].to(u.arcsec) / grid_step_ra))
        n_dec = int(np.abs(spherical_offset[1].to(u.arcsec) / grid_step_dec))

        if box.ra[0] > 180 * u.deg:
            print("... will wrap at RA = 0h such that grid spans -180 to +180 deg")
            box_ra = box.ra.deg
            box_ra[box_ra >= 180] -= 360
            box_dec = box.dec.deg
        else:
            box_ra = box.ra.deg
            box_dec = box.dec.deg

        grid_ra, grid_dec = np.meshgrid(
            np.linspace(box_ra[0], box_ra[1], n_ra),
            np.linspace(box_dec[0], box_dec[1], n_dec),
        )
        np.savez("grid",grid_ra,grid_dec)
        target_positions = SkyCoord(
            grid_ra,
            grid_dec,
            frame="icrs",
            unit=("deg", "deg"),
        )
        print(f"... target positions array shape, (nRA, nDec) = {n_ra, n_dec}")

    elif not args.position and args.gridbox is None:
        target_positions = form_grid_positions(
            look_positions[0],
            # max_separation_arcsec=(min(fwhm / 2)).to(u.arcsecond).value,
            max_separation_arcsec=(0.5 * u.arcmin).to(u.arcsecond).value,
            nlayers=args.nlayers,
            overlap=True,
        )
    else:
        for p in args.position.split(" "):
            target_ras.append(p.split("_")[0])
            target_decs.append(p.split("_")[1])

        target_positions = SkyCoord(
            target_ras,
            target_decs,
            frame="icrs",
            unit=("hourangle", "deg"),
        )
    t1 = timer.time()
    print(f"... took {t1-t0} seconds")

    print("Converting to AltAz...")
    t0 = timer.time()
    target_positions_altaz = target_positions.transform_to(altaz_frame)
    t1 = timer.time()
    print(f"... took {t1-t0} seconds")

    tabp_look = []
    afp_look = []
    for i, lp in enumerate(look_positions_altaz):
        print(
            "Processing look-direction = "
            f"{look_positions[i].ra.to_string(u.hour)} "
            f"{look_positions[i].dec.to_string(u.degree, alwayssign=True)}"
        )
        tabp_freq = []
        afp_freq = []
        for j, freq in enumerate(freqs):
            print(f"Processing frequency = {freq} Hz")
            print("Computing array factors...")
            t0 = timer.time()
            # Compute the array factor (tied-array beam weighting factor).
            look_psi = calcGeometricDelays(
                tile_positions,
                freq,
                lp.alt.rad,
                lp.az.rad,
            )
            target_psi = calcGeometricDelays(
                tile_positions,
                freq,
                target_positions_altaz.alt.rad,
                target_positions_altaz.az.rad,
            )
            afp = calcArrayFactorPower(look_psi, target_psi)
            t1 = timer.time()
            print(f"... took {t1-t0} seconds")

            if not args.nopb:
                # Compute the primary beam zenith-normalised power.
                print("Computing primary beam power...")
                t0 = timer.time()
                pbp = getPrimaryBeamPower(
                    context,
                    freq,
                    target_positions_altaz.alt.rad,
                    target_positions_altaz.az.rad,
                )
                print(f"... primary beam max. in-field power = {pbp.max():.3f}")
                t1 = timer.time()
                print(f"... took {t1-t0} seconds")
            else:
                pbp = 1.0

            # Finally, estimate the zenith-normalised tied-array beam power.
            if isinstance(pbp, float):
                tabp = afp
            else:
                tabp = afp * pbp.reshape(afp.shape)
            tabp_freq.append(tabp)
            afp_freq.append(afp)
        tabp_look.append(tabp_freq)
        afp_look.append(afp_freq)

    tabp_look = np.array(tabp_look)
    afp_look = np.array(afp_look)
    # print(f"afp shape = {afp_look.shape}")
    # print(f"tab shape = {tabp_look.shape}")

    if args.plot:
        if args.nopb:
            label = "Array factor power"
        else:
            label = "Zenith-normalised tied-array beam power"

        print("Plotting sky map...")

        # snr = [13, 26, 21, 10, 8, 8]
        # clip_mask = makeMaskFromWeightedPatterns(
        #     tabp_look.mean(axis=1), snr, snr_percentile=75
        # )

        # plt.imshow(
        #     clip_mask,
        #     aspect="auto",
        #     extent=[
        #         grid_ra.min(),
        #         grid_ra.max(),
        #         grid_dec.min(),
        #         grid_dec.max(),
        #     ],
        # )
        # plt.colorbar()
        # plt.contour(
        #     grid_ra,
        #     grid_dec,
        #     clip_mask,
        #     levels=np.percentile(clip_mask, [50, 80]),
        #     cmap=plt.get_cmap("Reds"),
        # )
        # plt.scatter(
        #     look_positions[0].ra.deg,
        #     look_positions[0].dec.deg,
        #     marker="x",
        #     c="r",
        # )
        # plt.errorbar(
        #     73.0029167,
        #     -34.3116667,
        #     xerr=0.00125,
        #     yerr=0.00111,
        #     marker="+",
        #     c="C1",
        # )
        # from matplotlib.patches import Circle

        # circ = Circle(
        #     xy=(73.0029167, -34.3116667),
        #     radius=(3 * u.arcmin).to(u.deg).value,
        #     facecolor="none",
        #     edgecolor="C1",
        # )
        # plt.gca().add_patch(circ)
        # plt.xlabel("Right Ascension (deg)")
        # plt.ylabel("Declination (deg)")
        # plt.show()
        # plt.savefig("mask.png", bbox_inches="tight", dpi=150)

        product = np.sum(tabp_look.mean(axis=1), axis=0)
        np.save('tabp_look',tabp_look)
        fig = plt.figure()
        ax = fig.add_subplot()
        ctr_levels = [0.01, 0.1, 0.25, 0.5, 0.8, 1]
        cmap = plt.get_cmap("Greys")

        map_extent = [
            grid_ra.min(),
            grid_ra.max(),
            grid_dec.min(),
            grid_dec.max(),
        ]

        tab_map = ax.imshow(
            product,
            aspect="auto",
            interpolation="none",
            # origin="lower",
            extent=map_extent,
            cmap=cmap,
            norm="log",
            vmin=min(ctr_levels),
            vmax=max(ctr_levels),
        )

        # if not args.nopb and len(freqs) == 1:
        #     pb_ctr = ax.contour(
        #         grid_ra,
        #         grid_dec,
        #         pbp.reshape(afp.shape),
        #         levels=6,
        #         colors="black",
        #         linestyles="dotted",
        #         norm="log",
        #     )
        for ld in tabp_look.mean(axis=1):
            tab_ctr = ax.contour(
                grid_ra,
                grid_dec,
                ld,
                levels=ctr_levels[2:-1],
                cmap="plasma",
                norm="log",
            )
        # ax.errorbar(
        #     73.0029167,
        #     -34.3116667,
        #     xerr=0.00125,
        #     yerr=0.00111,
        #     marker="+",
        #     c="C1",
        # )
        # tab_map = ax.tricontourf(
        #     target_positions.ra.deg,
        #     target_positions.dec.deg,
        #     product,
        #     levels=map_levels,
        #     extend="min",
        #     cmap=cmap,
        #     norm="log",
        #     vmin=map_levels[0],
        # )
        # tab_ctr = ax.tricontour(
        #     target_positions.ra.deg,
        #     target_positions.dec.deg,
        #     product,
        #     levels=ctr_levels[1:-1],
        #     cmap="plasma",
        #     norm="log",
        # )
        # ax.set_xlim(43, 47)
        # ax.set_ylim(-57, -53)
        # ax.set_xlim(20, 70)
        # ax.set_ylim(-70, -40)
        ax.set_xlabel("Right Ascension (deg)", fontsize=14)
        ax.set_ylabel("Declination (deg)", fontsize=14)
        ax.tick_params(labelsize=12)

        # values under the minimum aren't plotted, so revert to the default figure facecolor
        # ax.set_facecolor(cmap(map_levels[0]))
        cbar = plt.colorbar(
            tab_map,
            ticks=ctr_levels,
            format=mticker.ScalarFormatter(),
            extend="min",
        )
        cbar.add_lines(tab_ctr)
        # if not args.nopb and len(freqs) == 1:
        #     ax.clabel(pb_ctr, fmt="%g", fontsize=10)
        cbar.set_label(fontsize=12, label=label)
        cbar.ax.tick_params(labelsize=11)

        oname_base = f"{context.obs_id}_tiedarray_beam"

        if args.nopb:
            oname_base += "_nopb"
        else:
            oname_base += "_pb"

        if len(args.freq) > 1:
            oname_base += "_multifreq"

        plt.savefig(f"{oname_base}.png", dpi=200, bbox_inches="tight")

        if not args.nopb and len(freqs) == 1:
            print("Plotting beam slices...")
            fig.clear()
            ax = fig.add_subplot()
            pb_map = ax.imshow(
                pbp.reshape(afp.shape),
                aspect="auto",
                interpolation="none",
                # origin="lower",
                extent=map_extent,
                cmap=cmap,
                # norm="log",
                # vmin=min(ctr_levels),
                # vmax=max(ctr_levels),
            )
            cbar = plt.colorbar(
                pb_map,
                # ticks=ctr_levels,
                # format=mticker.ScalarFormatter(),
                # extend="min",
            )
            # cbar.add_lines(tab_ctr)
            cbar.set_label(fontsize=12, label="Zenith-normalised primary beam power")
            cbar.ax.tick_params(labelsize=11)

            ax.set_xlabel("Right Ascension (deg)", fontsize=14)
            ax.set_ylabel("Declination (deg)", fontsize=14)
            ax.tick_params(labelsize=12)
            plt.savefig(f"{context.obs_id}_field_pb.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            fig, (ax1, ax2) = plt.subplots(2, 1)

            ra_trace_afp = np.prod(afp_look.mean(axis=1), axis=0).mean(axis=0)
            ra_trace_tab = product.mean(axis=0)
            dec_trace_afp = np.prod(afp_look.mean(axis=1), axis=0).mean(axis=1)
            dec_trace_tab = product.mean(axis=1)
            # ax1.plot(
            #     np.linspace(box_ra[0], box_ra[1], n_ra),
            #     ra_trace_afp,
            #     label="AFP",
            # )
            # ax1.plot(
            #     np.linspace(box_ra[0], box_ra[1], n_ra),
            #     (ra_trace_tab / ra_trace_tab.max()) * ra_trace_afp.max(),
            #     label="TAB (Scaled)",
            #     ls="--",
            # )
            dec_linear = np.linspace(box_dec[0], box_dec[1], n_dec)
            ra_linear = np.linspace(box_ra[0], box_ra[1], n_ra)
            ax1.plot(
                ra_linear,
                product[n_dec // 2, :],
                label=f"Slice (dec={dec_linear[n_dec // 2]:g}d)",
                color="C0",
                ls="-",
            )
            ax1.plot(
                ra_linear,
                product[n_dec // 2 - 50, :],
                label=f"Slice (dec={dec_linear[n_dec // 2 - 50]:g}d)",
                color="C1",
                ls="-",
            )
            # ax1.plot(
            #     ra_linear,
            #     product[n_dec // 2 + 50, :],
            #     label=f"Slice (dec={dec_linear[n_dec // 2 + 50]:g}d)",
            #     color="C1",
            #     ls="--",
            # )
            ax1.plot(
                ra_linear,
                product[n_dec // 2 - 100, :],
                label=f"Slice (dec={dec_linear[n_dec // 2 - 100]:g}d)",
                color="C2",
                ls="-",
            )
            # ax1.plot(
            #     ra_linear,
            #     product[n_dec // 2 + 100, :],
            #     label=f"Slice (dec={dec_linear[n_dec // 2 + 100]:g}d)",
            #     color="C2",
            #     ls="--",
            # )

            # ax2.plot(
            #     dec_linear,
            #     dec_trace_afp,
            #     label="AFP",
            # )
            # ax2.plot(
            #     dec_linear,
            #     (dec_trace_tab / dec_trace_tab.max()) * dec_trace_afp.max(),
            #     label="TAB (Scaled)",
            #     ls="--",
            # )
            ax2.plot(
                dec_linear,
                product[:, n_ra // 2],
                label=f"Slice (ra={ra_linear[n_ra // 2]:g}d)",
                color="C0",
                ls="-",
            )
            ax2.plot(
                dec_linear,
                product[:, n_ra // 2 - 50],
                label=f"Slice (ra={ra_linear[n_ra // 2 - 50]:g}d)",
                color="C1",
                ls="-",
            )
            # ax2.plot(
            #     dec_linear,
            #     product[:, n_ra // 2 + 50],
            #     label=f"Slice (ra={ra_linear[n_ra // 2 + 50]:g}d)",
            #     color="C1",
            #     ls="--",
            # )
            ax2.plot(
                dec_linear,
                product[:, n_ra // 2 - 100],
                label=f"Slice (ra={ra_linear[n_ra // 2 - 100]:g}d)",
                color="C2",
                ls="-",
            )
            # ax2.plot(
            #     dec_linear,
            #     product[:, n_ra // 2 + 100],
            #     label=f"Slice (ra={ra_linear[n_ra // 2 + 100]:g}d)",
            #     color="C2",
            #     ls="--",
            # )
            ax1.set_xlabel("Right Ascension (deg)")
            ax2.set_xlabel("Declination (deg)")
            for ax in (ax1, ax2):
                # ax.set_ylim(0, None)
                ax.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(f"{context.obs_id}_pattern_cuts.png")

    tt1 = timer.time()
    print(f"Done!! (Took {tt1-tt0} seconds.)\n")


if __name__ == "__main__":
    main()
