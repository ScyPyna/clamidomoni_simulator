"""The first 2 minutes of the 28-minute runs against the window used so far (2026-10-26).

The lab records 2-minute videos at 10 fps; every lab comparison so far used the last 419 s of
the 28-minute simulations (pulse_ratio.py, radial_profiles.py, lab_fit.py). This reads only the
head of each log, up to t = 120 s, and computes the same rho_in/rho_out as pulse_ratio.py:
  early_end  mean over 100 <= t <= 120 s (the end of a 2-minute acquisition)
  early_all  mean over 0 <= t <= 120 s
next to ratio_window (last 419 s) from pulse_ratio_cache.csv. Cache: figs/early_cache.csv.
"""
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/media/user/LaCie/pulsed_exps")
CACHE = OUT/"figs"/"early_cache.csv"
LATE = OUT/"figs"/"pulse_ratio_cache.csv"
MAP = Path("/home/user/Scrivania/pulsed_simulation/tbr_fields/tbr_map_ring_180muW.dat")
PART_DIAM = 10.0
NPART = 371
T_ACQ = 120.0
FRAME = 2**13*1e-4          # 0.8192 s between logged frames at Nlevel 24
NFRAMES = int(T_ACQ/FRAME) + 2


def ring_geometry(mapfile):
    """Box side and radius of the tumbling-rate maximum (as pulse_ratio.py), in microns."""
    with open(mapfile) as f:
        Nx, _, _, _, pix = f.readline().split()
        Nx, pix = int(Nx), float(pix)
        m = np.loadtxt(f)
    i = (np.arange(Nx) - Nx/2 + 0.5)*pix
    X, Y = np.meshgrid(i, i, indexing="ij")
    r = np.sqrt(X*X + Y*Y)
    edges = np.arange(0, Nx*pix/2, 2.0)
    prof = np.array([m[(r >= a) & (r < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    return Nx*pix, 0.5*(edges[1:] + edges[:-1])[np.argmax(prof)]


def early(run, L, r_ring):
    log = next((OUT/run/"logs").glob("Simlog*"))
    d = pd.read_csv(log, sep=" ", skiprows=2, header=None, usecols=[0, 2, 3], names=["t", "x", "y"],
                    nrows=NFRAMES*NPART)
    d = d[d.t <= T_ACQ + 1e-9]
    r = np.hypot(d.x, d.y)*PART_DIAM
    n_in = (r < r_ring).groupby(d.t).sum()
    a_in = np.pi*r_ring**2
    ratio = (n_in/a_in)/np.maximum(NPART - n_in, 1)*(L*L - a_in)
    t = ratio.index.to_numpy()
    return ratio[t >= 100].mean(), ratio.mean(), len(t)


def early_profile(run, t_from=90.0):
    """Radial counts (radial_profiles.py bins) over light-on frames with t_from <= t <= 120 s."""
    import radial_profiles as rp
    log = next((OUT/run/"logs").glob("Simlog*"))
    d = pd.read_csv(log, sep=" ", skiprows=2, header=None, usecols=[0, 2, 3, 7],
                    names=["t", "x", "y", "light"], nrows=NFRAMES*NPART)
    d = d[(d.t >= t_from) & (d.t <= T_ACQ + 1e-9) & (d.light > 0.5)]
    h, _ = np.histogram(np.hypot(d.x, d.y)*PART_DIAM, bins=rp.EDGES)
    return h, d.t.nunique()


def profiles(runs):
    """Early radial counts for the given runs, cached in figs/early_radial_cache.csv."""
    import radial_profiles as rp
    path = OUT/"figs"/"early_radial_cache.csv"
    cache = {}
    if path.exists():
        for row in csv.reader(open(path)):
            if row[0] != "run":
                cache[row[0]] = (np.array(row[2:], dtype=float), int(row[1]))
    for run in runs:
        if run not in cache and (OUT/run).exists():
            cache[run] = early_profile(run)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "frames"] + [f"{a:.0f}-{b:.0f}" for a, b in zip(rp.EDGES[:-1], rp.EDGES[1:])])
        for k, (h, n) in sorted(cache.items()):
            w.writerow([k, n] + list(np.asarray(h).astype(int)))
    return cache


if __name__ == "__main__":
    L, r_ring = ring_geometry(MAP)
    done = set(pd.read_csv(CACHE).run) if CACHE.exists() else set()
    runs = [r for r in pd.read_csv(LATE).run if r not in done]
    new = not CACHE.exists()
    with open(CACHE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "early_end", "early_all", "frames"])
        for k, run in enumerate(runs):
            e_end, e_all, n = early(run, L, r_ring)
            w.writerow([run, f"{e_end:.6f}", f"{e_all:.6f}", n])
            if k % 100 == 0:
                f.flush()
                print(k, run, flush=True)
    print("done", len(runs))
