"""Light and tumbling-rate maps built from the lab's illumination images (2026-09-29).

One map per acquisition session, from /home/user/Scaricati/illuminationTiffs, scaled to the power
of that session. It replaces helper_scripts/ring_profile_analyzer.py of the fork for this use,
fixing three things that bias exactly the quantities that matter here:
  - the ring centre is found from the image (intensity-weighted centroid of the bright ring),
    not taken as the image centre (rings are off-centre by up to ~80 px);
  - the camera background is subtracted before normalising, so it does not absorb power;
  - the radial profile averages all angles, not only +-45 degrees on one side.

The output keeps the simulator's format and grid: 1159 x 1159 px of 1.041667 um (box 1207 um,
371 cells at rho 0.02, as in every previous run), light in uW/um^2, tumbling rate from
inputSimulations/tumblingVsEnergyDensity.txt exactly as in ill_maker.py.

Files: tbr_fields_lab/ill<S>/{ill,tbr}_map_ring_<P>muW.dat, where P is the session power;
set power_input to P in the yaml and point the run's tbr_fields at that folder.
The maps stay on the system disk: every run reads them at start-up, and 48 runs starting
together on the external drive would make it seek-bound.
"""
import json
from pathlib import Path
import numpy as np
from matplotlib import image
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

BASE  = Path("/home/user/Scrivania/pulsed_simulation")
TIFFS = Path("/home/user/Scaricati/illuminationTiffs")
TUMB  = Path(__file__).resolve().parent/"tumblingVsEnergyDensity.txt"   # calibration file next to this script
OUTD  = BASE/"tbr_fields_lab"
PIX   = 1.041667            # um per pixel, camera and simulation grid
NRAD  = 821                 # radial samples: gives the same 1159 x 1159 map as ring_180muW
SMOOTH_PX = 2.0             # light smoothing of the 8-bit radial profile
THR   = 3.0708352           # working reversal threshold, only for the printed summary

# session -> (image, power in uW); session 104 used exactly the illumination and power of 103,
# so it has no map of its own: its runs use ill103
SESSIONS = {98: ("illumination98.tiff", 150), 99: ("illumination99.tiff", 180),
            100: ("Illumination100.tiff", 140), 101: ("illumination101.tiff", 140),
            102: ("illumination102.tiff", 140), 103: ("illumination103.tiff", 180),
            105: ("illumination105.tiff", 180)}


def radial_profile(path):
    im = image.imread(path).astype(float)
    if im.ndim == 3:
        im = im[..., 0]
    bg = np.percentile(im, 5)
    a = np.clip(im - bg, 0, None)
    bright = a > 0.5*a.max()
    yy, xx = np.indices(a.shape)
    cy = (yy[bright]*a[bright]).sum()/a[bright].sum()
    cx = (xx[bright]*a[bright]).sum()/a[bright].sum()
    rpx = np.hypot(yy - cy, xx - cx)
    k = np.floor(rpx).astype(int)
    s = np.bincount(k.ravel(), a.ravel()); n = np.bincount(k.ravel())
    prof = s/np.maximum(n, 1)
    # keep radii where at least a quarter of the circle lies inside the image
    full = 2*np.pi*(np.arange(len(n)) + 0.5)
    ok = n >= 0.25*full
    last = np.argmax(~ok[1:]) if (~ok[1:]).any() else len(ok) - 1
    prof = gaussian_filter1d(prof[:last], SMOOTH_PX)
    # beyond the image: exponential continuation of the outer tail, fitted on its last 100 px
    r = np.arange(len(prof), dtype=float)
    tail = slice(max(0, len(prof) - 100), len(prof))
    y = np.clip(prof[tail], 1e-12, None)
    slope, icpt = np.polyfit(r[tail], np.log(y), 1)
    slope = min(slope, -1e-3)
    ext = np.arange(len(prof), NRAD, dtype=float)
    full_prof = np.concatenate([prof, np.exp(icpt + slope*ext)])[:NRAD]
    info = dict(centre_px=[round(cx, 1), round(cy, 1)], background=round(bg, 1),
                data_radius_um=round(last*PIX, 1), saturated_pct=round(100*(im >= 255).mean(), 2))
    return full_prof, info


def to_energy_density(prof, power):
    r = np.arange(len(prof))*PIX
    return prof*power/np.trapezoid(2*np.pi*r*prof, r)


td = np.genfromtxt(TUMB, skip_header=1)
tumble_converter = interp1d(td[:, 2], td[:, 1], kind="slinear", fill_value="extrapolate")


def write_maps(E, folder, power):
    """2D maps as in ill_maker.py (cubic radial interpolation, zero beyond the profile)."""
    conv = interp1d(np.arange(len(E))*PIX, E, kind="cubic", fill_value=0, bounds_error=False)
    pix_L = int(len(E)/np.sqrt(2)) - 1
    ix = np.arange(-pix_L, pix_L + 1)*PIX
    X, Y = np.meshgrid(ix, ix)
    ill = np.clip(conv(np.sqrt(X**2 + Y**2)), 0, None)
    tbr = tumble_converter(ill)
    Nx = Ny = len(ix)
    folder.mkdir(parents=True, exist_ok=True)
    for kind, m in (("ill", ill), ("tbr", tbr)):
        with open(folder/f"{kind}_map_ring_{power}muW.dat", "w") as f:
            f.write(f"{Nx} {Ny} {ill.mean()} {tbr.mean()} {PIX}\n")
            np.savetxt(f, m)
    return Nx, tbr


if __name__ == "__main__":
    summary = {}
    r = np.arange(NRAD)*PIX
    for s, (fname, power) in SESSIONS.items():
        prof, info = radial_profile(TIFFS/fname)
        E = to_energy_density(prof, power)
        Nx, tbr = write_maps(E, OUTD/f"ill{s}", power)
        t = tumble_converter(E)
        above = r[t > THR]
        k = E.argmax(); h = E[k]/2
        lo = r[:k][E[:k] < h]; hi = r[k:][E[k:] < h]
        info.update(image=fname, power_uW=power, grid=Nx, peak_um=round(r[k], 1),
                    half_max_um=[round(lo.max() if len(lo) else 0, 1), round(hi.min(), 1)],
                    peak_E=float(E[k]), centre_E=float(E[0]), centre_tumbling=round(float(t[0]), 3),
                    above_threshold_um=[round(above.min(), 1), round(above.max(), 1)] if len(above) else None)
        np.savetxt(OUTD/f"ill{s}"/"radial_energy_density.txt", E, header="uW/um^2 vs radius, 1.041667 um steps")
        summary[s] = info
        print(s, info)
    json.dump(summary, open(OUTD/"summary.json", "w"), indent=1)
