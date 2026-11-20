"""
Generate speckle illumination / tumbling-rate maps for the Clamidomoni simulator.

Replaces ill_maker.py for the speckle experiments. The difference with respect to
speckle_maker.py is that the field is produced directly on the simulation grid
(same pixel size as the ring maps), is exactly periodic by construction, and is
normalised to a target mean energy density in muW/mum^2 so that
tumblingVsEnergyDensity.txt is used inside its calibrated range.

The field is built in Fourier space: complex white noise on a disk of radius a
(in FFT index units), inverse-transformed to give the complex amplitude E, with
I = |E|^2. This is fully developed speckle, so the one-point distribution is
exponential whatever the grain size, and only the two-point correlation changes
with a. Grain size is defined as the FWHM of the intensity autocorrelation and
follows xi_px = 0.5145 * N / a for a circular pupil.
"""

import argparse
import json
import os

import numpy as np
from scipy.interpolate import interp1d


PIX = 1.041667                      # mum per pixel, same as the ring maps
CALIB = "inputSimulations//tumblingVsEnergyDensity.txt"
AIRY_FWHM = 0.5145                  # FWHM of |2 J1(u)/u|^2 in units of N/a
GRAD_SIGN_THRESH = 3.0708352        # tbr at which the gradient response reverses


# -----------------------------------------------------------------------------
# Speckle field
# -----------------------------------------------------------------------------
def speckle_field(n_pix, grain_px, rng):
    """Periodic fully developed speckle on an n_pix x n_pix grid."""
    a = AIRY_FWHM * n_pix / grain_px
    if a < 1.5:
        raise ValueError(
            f"grain {grain_px:.1f} px too large for a {n_pix} px box "
            f"(pupil radius {a:.2f} px, too few modes)."
        )

    k = np.fft.fftfreq(n_pix) * n_pix
    kx, ky = np.meshgrid(k, k, indexing="ij")
    pupil = (kx**2 + ky**2) <= a**2

    amp = (rng.normal(size=(n_pix, n_pix)) + 1j * rng.normal(size=(n_pix, n_pix)))
    field = np.fft.ifft2(amp * pupil)
    return np.abs(field) ** 2


def measure_grain(intensity, pix, max_lag_px):
    """FWHM of the intensity autocorrelation, in mum.

    Uses the FFT of the fluctuation field, so it is exact for a periodic map and
    needs no windowing. Returns the radially averaged FWHM.
    """
    fluct = intensity - intensity.mean()
    power = np.abs(np.fft.fft2(fluct)) ** 2
    auto = np.fft.fftshift(np.real(np.fft.ifft2(power)))
    auto /= auto.max()

    n = intensity.shape[0]
    c = n // 2
    lag = int(min(max_lag_px, c - 1))
    sub = auto[c - lag:c + lag + 1, c - lag:c + lag + 1]

    yy, xx = np.mgrid[-lag:lag + 1, -lag:lag + 1]
    r = np.sqrt(xx**2 + yy**2).ravel()
    v = sub.ravel()
    order = np.argsort(r)
    r, v = r[order], v[order]

    # radial average on 1-pixel bins, then linear crossing of the 0.5 level
    bins = np.arange(0, lag + 1)
    idx = np.digitize(r, bins) - 1
    prof = np.array([v[idx == i].mean() if np.any(idx == i) else np.nan
                     for i in range(len(bins))])
    good = ~np.isnan(prof)
    bins, prof = bins[good], prof[good]

    below = np.where(prof < 0.5)[0]
    if len(below) == 0:
        return np.nan
    j = below[0]
    r_half = np.interp(0.5, [prof[j], prof[j - 1]], [bins[j], bins[j - 1]])
    return 2 * r_half * pix


# -----------------------------------------------------------------------------
# Maps
# -----------------------------------------------------------------------------
def write_map(path, field, pix, avg_int, avg_tbr):
    with open(path, "w") as f:
        f.write(f"{field.shape[0]} {field.shape[1]} {avg_int} {avg_tbr} {pix}\n")
    with open(path, "a") as f:
        np.savetxt(f, field)


def build(grain_um, n_pix, mean_int, realisation, outdir, homogeneous=False,
          match_peak=None):
    os.makedirs(outdir, exist_ok=True)

    tumble_data = np.genfromtxt(CALIB, skip_header=1)
    tumble_converter = interp1d(tumble_data[:, 2], tumble_data[:, 1],
                                kind="slinear", fill_value="extrapolate")
    calib_max = tumble_data[:, 2].max()

    if homogeneous:
        ill_map = np.full((n_pix, n_pix), mean_int)
        grain_meas = np.nan
    else:
        # explicit reproducible seed: grain size in units of 0.1 um, plus the
        # realisation index. Same (grain, realisation) always gives the same map.
        rng = np.random.default_rng(int(round(grain_um * 10)) * 1000 + realisation)
        ill_map = speckle_field(n_pix, grain_um / PIX, rng)
        if match_peak is None:
            ill_map *= mean_int / ill_map.mean()
        else:
            # Experimental criterion: speckle peak equal to the ring peak. Note
            # this makes the *mean* drift with grain size, because the peak of an
            # exponential field scales as <I> ln(N_grains).
            ill_map *= match_peak / ill_map.max()
        grain_meas = measure_grain(ill_map, PIX, max_lag_px=4 * grain_um / PIX)

    tbr_map = tumble_converter(ill_map)

    stats = {
        "grain_target_um": None if homogeneous else grain_um,
        "grain_measured_um": None if homogeneous else round(float(grain_meas), 2),
        "realisation": realisation,
        "n_pix": n_pix,
        "box_um": round(n_pix * PIX, 1),
        "grains_per_side": None if homogeneous else round(n_pix * PIX / grain_um, 1),
        "I_mean": float(ill_map.mean()),
        "I_max": float(ill_map.max()),
        "frac_above_calib": float((ill_map > calib_max).mean()),
        "tbr_min": float(tbr_map.min()),
        "tbr_mean": float(tbr_map.mean()),
        "tbr_max": float(tbr_map.max()),
        "frac_above_thresh": float((tbr_map > GRAD_SIGN_THRESH).mean()),
    }

    return ill_map, tbr_map, stats


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grains", type=float, nargs="+",
                   default=[5, 10, 20, 40, 80, 160],
                   help="grain sizes in mum (FWHM of intensity autocorrelation)")
    p.add_argument("--npix", type=int, default=1159)
    p.add_argument("--mean-int", type=float, default=3.4e-5,
                   help="target mean energy density in muW/mum^2")
    p.add_argument("--realisations", type=int, default=3)
    p.add_argument("--match-peak", type=float, default=None,
                   help="instead of matching the mean, scale each map so that its "
                        "peak equals this energy density (ring 180 muW peak is "
                        "1.09e-3). Makes the mean drift with grain size.")
    p.add_argument("--outroot", default="maps")
    p.add_argument("--homogeneous", action="store_true",
                   help="also build the flat control map")
    p.add_argument("--dry-run", action="store_true",
                   help="compute and report statistics without writing the maps")
    args = p.parse_args()

    jobs = [(g, m) for g in args.grains for m in range(args.realisations)]
    if args.homogeneous:
        jobs.append((None, 0))

    all_stats = []
    for grain, m in jobs:
        hmg = grain is None
        ill, tbr, st = build(grain, args.npix, args.mean_int, m,
                             args.outroot, homogeneous=hmg,
                             match_peak=args.match_peak)

        if hmg:
            tag, power_id = "hmg", int(round(args.mean_int * 1e7))
            outdir = os.path.join(args.outroot, "hmg")
            name_i = f"ill_map_hmg_{power_id}muW.dat"
            name_t = f"tbr_map_hmg_{power_id}muW.dat"
        else:
            power_id = int(round(grain))
            outdir = os.path.join(args.outroot, f"xi{power_id}_m{m}")
            name_i = f"ill_map_ring_{power_id}muW.dat"
            name_t = f"tbr_map_ring_{power_id}muW.dat"
            tag = f"xi{power_id}_m{m}"

        st["tag"] = tag
        st["power_id"] = power_id
        all_stats.append(st)

        if not args.dry_run:
            os.makedirs(outdir, exist_ok=True)
            # read_field() is called for both the tbr and the ill map regardless
            # of S_model, so both .dat must exist or the ill field is left with
            # Nx = pixel_size = 0. The .npy duplicate is what the analysis reads
            # to get I at the particle positions.
            write_map(os.path.join(outdir, name_t), tbr, PIX,
                      st["I_mean"], st["tbr_mean"])
            write_map(os.path.join(outdir, name_i), ill, PIX,
                      st["I_mean"], st["tbr_mean"])
            np.save(os.path.join(outdir, "ill_map.npy"), ill.astype(np.float32))

        print(f"{tag:12s} grain={str(st['grain_measured_um']):>6s}um "
              f"n_grains/side={str(st['grains_per_side']):>5s} "
              f"I_max={st['I_max']:.3g} tbr=[{st['tbr_min']:.3f},"
              f"{st['tbr_max']:.3f}] mean={st['tbr_mean']:.3f} "
              f"above_thresh={st['frac_above_thresh']*100:5.2f}% "
              f"outside_calib={st['frac_above_calib']:.1e}")

    with open(os.path.join(args.outroot, "map_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)


if __name__ == "__main__":
    main()
