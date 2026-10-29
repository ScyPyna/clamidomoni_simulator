"""P(theta) from velocities, computed the way the lab pipeline does, on the high-resolution runs.

Lab recipe (package_GV/video_test/clamutils_exp.py, grad_theta, particle_show = 5, dt = 0.1 s):
  V        = (r[t+w] - r[t]) / (w dt)                    finite difference over w frames
  grad I   = moving average of the sampled gradient over the same w+1 frames
  theta    = arccos( V . grad I / (|V| |grad I|) ),      then every w-th value (non-overlapping)
  inside   = r <= 300 um, outside = 300 < r < 800 um, 10 bins, density=True

The high-resolution runs log every 0.1024 s (print_skip_exp 10), so w = 5 is 0.512 s.
theta is also computed from the instantaneous orientation phi (the earlier analysis), and for
w = 1, 2, 5, 10 frames to show how the window changes the answer. Samples are split by the
reversal state (majority over the window) and by the light state.

Per-run histograms are cached in figs/theta_vel_cache/<run>.npz.
"""
import glob, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

OUT   = Path("/media/user/LaCie/pulsed_exps")
FIGS  = OUT/"figs"
CACHE = FIGS/"theta_vel_cache"
ILL   = Path("/home/user/Scrivania/pulsed_simulation/tbr_fields/ill_map_ring_180muW.dat")
N     = 371
T_SKIP = 419.0                # s, transient left out (same 25 % as the density analysis)
WINDOWS = [1, 2, 5, 10]       # frames; 5 = the lab's particle_show
LAB_W = 5
R_IN, R_OUT = 300.0, 800.0
EDGES = np.linspace(0, np.pi, 11)          # the lab's 10 bins
REGIONS = ("in", "out")
STATES = ("all", "norm", "rev")
LIGHTS = ("all", "on", "off")


def load_map():
    with open(ILL) as f:
        Nx, Ny, _, _, pix = f.readline().split()
    m = pd.read_csv(ILL, sep=r"\s+", header=None, skiprows=1, engine="c").values
    gx, gy = np.gradient(m, float(pix))
    return gx, gy, int(Nx), int(Ny), float(pix)


G = None


def grad_at(x, y):
    gx, gy, Nx, Ny, pix = G
    ix = np.clip(((x + Nx*pix/2)/pix).astype(int), 0, Nx-1)
    iy = np.clip(((y + Ny*pix/2)/pix).astype(int), 0, Ny-1)
    return gx[ix, iy], gy[ix, iy]


def movavg(a, w):
    """Moving average over w points along axis 0 (valid part), like np.convolve(..., 'valid')/w."""
    c = np.cumsum(a, axis=0, dtype=float)
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), c])
    return (c[w:] - c[:-w])/w


def hist_block(cos, region, rev, light):
    """Counts for every (region, state, light) combination, plus sum of cos."""
    th = np.arccos(np.clip(cos, -1, 1))
    out = {}
    for rg in REGIONS:
        mr = region == rg
        for st in STATES:
            ms = mr if st == "all" else (mr & (rev if st == "rev" else ~rev))
            for lt in LIGHTS:
                m = ms if lt == "all" else (ms & (light if lt == "on" else ~light))
                h, _ = np.histogram(th[m], bins=EDGES)
                out[(rg, st, lt)] = (h, cos[m].sum(), m.sum())
    return out


def pack(prefix, res, store):
    for (rg, st, lt), (h, s, n) in res.items():
        k = f"{prefix}|{rg}|{st}|{lt}"
        store[k+"|h"] = h; store[k+"|s"] = s; store[k+"|n"] = n


def process(run):
    dst = CACHE/f"{run}.npz"
    if dst.exists():
        return run, "cached"
    f = glob.glob(str(OUT/run/"logs"/"Simlog*"))[0]
    d = pd.read_csv(f, sep=" ", header=None, skiprows=2, engine="c").values
    nf = len(d)//N
    d = d[:nf*N].reshape(nf, N, -1)
    t = d[:, 0, 0]
    d = d[t >= T_SKIP]
    x, y = d[:, :, 2]*10, d[:, :, 3]*10
    phi, state, light = d[:, :, 4], d[:, :, 6], d[:, :, 7]
    L = G[2]*G[4]
    gx, gy = grad_at(x, y)
    r = np.hypot(x, y)
    store = {}

    # instantaneous orientation, every frame
    gn = np.hypot(gx, gy); ok = gn > 0
    cos = (np.cos(phi)*gx + np.sin(phi)*gy)[ok]/gn[ok]
    reg = np.where(r[ok] <= R_IN, "in", np.where(r[ok] < R_OUT, "out", "none"))
    pack("inst", hist_block(cos, reg, state[ok] < 0, light[ok] > 0.5), store)

    for w in WINDOWS:
        dx = x[w:] - x[:-w]; dy = y[w:] - y[:-w]
        dx -= L*np.round(dx/L); dy -= L*np.round(dy/L)          # periodic box
        gxa, gya = movavg(gx, w+1), movavg(gy, w+1)
        ra = movavg(r, w+1)
        sa = movavg(state, w+1); la = movavg(light, w+1)
        # every w-th window, as the lab does ([::particle_show])
        sl = slice(None, None, w)
        dx, dy, gxa, gya, ra, sa, la = dx[sl], dy[sl], gxa[sl], gya[sl], ra[sl], sa[sl], la[sl]
        v = np.hypot(dx, dy); g = np.hypot(gxa, gya)
        ok = (v > 0) & (g > 0)
        cos = (dx*gxa + dy*gya)[ok]/(v[ok]*g[ok])
        reg = np.where(ra[ok] <= R_IN, "in", np.where(ra[ok] < R_OUT, "out", "none"))
        pack(f"w{w}", hist_block(cos, reg, sa[ok] < 0, la[ok] > 0.5), store)
    np.savez(dst, **store)
    return run, f"{nf} frames"


def init():
    global G
    G = load_map()


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    runs = sorted(os.path.basename(p) for p in glob.glob(str(OUT/"hires_*")))
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    with Pool(nproc, initializer=init) as pool:
        for run, msg in pool.imap_unordered(process, runs):
            print(run, msg, flush=True)
