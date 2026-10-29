"""Density ratio inside/outside the ring for the pulsed-light runs.

rho_in  = N(r <  r_ring) / (pi r_ring^2)
rho_out = N(r >= r_ring) / (L^2 - pi r_ring^2)

r_ring is the radius where the tumbling-rate map peaks (the crest of the light
ring), L the side of the periodic box. A ratio of 1 means a uniform distribution,
above 1 accumulation inside the ring, below 1 depletion.

Each point of the grid is run with SEEDS independent seeds, mirroring the five
video acquisitions taken per point in the lab, so the error bar is the seed-to-seed
scatter and not an internal estimate. Two variants are produced: the value on the
final logged frame, and the average over the last WINDOW frames — the second is the
steadier per-seed estimator.

Per-run numbers are cached in figs/pulse_ratio_cache.csv, so re-plotting does not
re-read the (large, external-disk) logs. Delete the cache to force a recompute.
"""
import numpy as np, subprocess, csv, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_lang  # noqa: F401  Italian labels when FIG_LANG=it

OUT       = Path("/media/user/LaCie/pulsed_exps")
FIGS      = OUT/"figs"
CACHE     = FIGS/"pulse_ratio_cache.csv"
FIGS_EN   = FIGS/"en"     # figures (English labels); cache and csv stay in FIGS
MAP       = Path("tbr_fields/tbr_map_ring_180muW.dat")
PART_DIAM = 10.0          # microns, positions in the log are in particle diameters
TAUS      = [4.0, 8.0, 16.0, 20.0]
ONS       = [1, 5, 10, 15, 20]
OFFS      = [0.01, 0.1, 0.5, 1, 2, 5, 7, 10, 15, 20]
SEEDS     = [0, 1, 2, 3, 4]
WINDOW    = 512           # logged frames averaged at the end of the run
NLEVEL    = 24            # as in the runs; with PRINT_SKIP it gives the expected log length
PRINT_SKIP = 2**(7+24-18)
ERRBAR    = "std"         # 'std' = seed-to-seed spread (comparable to the lab replicas), 'sem' = error on the mean
# ordinal blue ramp, light->dark = short->long ON time (validated: monotone L, single hue)
COLORS    = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def ring_geometry(mapfile):
    """Box side and radius of the tumbling-rate maximum, both in microns."""
    with open(mapfile) as f:
        Nx, Ny, _, _, pix = f.readline().split()
        Nx, pix = int(Nx), float(pix)
        m = np.loadtxt(f)
    i = (np.arange(Nx) - Nx/2 + 0.5)*pix
    X, Y = np.meshgrid(i, i, indexing="ij")
    r = np.sqrt(X*X + Y*Y)
    edges = np.arange(0, Nx*pix/2, 2.0)
    prof = np.array([m[(r >= a) & (r < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    return Nx*pix, 0.5*(edges[1:] + edges[:-1])[np.argmax(prof)]


def run_values(run, L, r_ring):
    """(ratio on the last frame, ratio averaged over the final window) or None."""
    logs = OUT/run/"logs"
    if not logs.is_dir():
        return None
    try:
        f = next(logs.glob("Simlog*"))
    except StopIteration:
        return None
    with open(f) as fh:                       # particles per frame, from the first block
        fh.readline(); fh.readline()
        t0 = fh.readline().split()[0]
        npart = 1
        for line in fh:
            if line.split()[0] != t0:
                break
            npart += 1
    # a run still writing its log would be read truncated and the wrong value would be
    # cached for good, so only accept a log with every frame present
    nlines = int(subprocess.run(["wc", "-l", str(f)], capture_output=True,
                                text=True).stdout.split()[0])
    nframes_exp = 2**NLEVEL//PRINT_SKIP + 1
    if nlines != 2 + nframes_exp*npart:
        return None
    tail = subprocess.run(["tail", "-n", str(npart*WINDOW), str(f)],
                          capture_output=True, text=True).stdout.splitlines()
    if len(tail) < npart:
        return None
    d = np.loadtxt(tail[len(tail) % npart:])  # drop a partial leading frame
    nframes = len(d)//npart
    x = d[:, 2].reshape(nframes, npart)*PART_DIAM
    y = d[:, 3].reshape(nframes, npart)*PART_DIAM
    r = np.sqrt(x*x + y*y)

    a_in  = np.pi*r_ring**2
    a_out = L*L - a_in
    n_in  = (r < r_ring).sum(axis=1)
    ratio = (n_in/a_in)/np.maximum(npart - n_in, 1)*a_out
    return ratio[-1], ratio.mean()


L, r_ring = ring_geometry(MAP)
print(f"box L = {L:.1f} um, r_ring = {r_ring:.1f} um, area_in/area_box = {np.pi*r_ring**2/L**2:.3f}")

cache = {}
if CACHE.exists():
    for row in csv.DictReader(open(CACHE)):
        cache[row["run"]] = (float(row["ratio_tend"]), float(row["ratio_window"]))

missing = []
for tau in TAUS:
    for on in ONS:
        for off in OFFS:
            for s in SEEDS:
                run = f"tau{tau:g}_on{on:g}_off{off:g}_seed{s}"
                if run in cache:
                    continue
                v = run_values(run, L, r_ring)
                if v is None:
                    missing.append(run)
                else:
                    cache[run] = v
if missing:
    print(f"ATTENZIONE: {len(missing)} run non ancora disponibili, es. {missing[:3]}")

FIGS.mkdir(exist_ok=True)
with open(CACHE, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run", "ratio_tend", "ratio_window"])
    for k in sorted(cache): w.writerow([k, f"{cache[k][0]:.6f}", f"{cache[k][1]:.6f}"])

# aggregate over seeds
agg, rows = {}, []
for tau in TAUS:
    for on in ONS:
        for off in OFFS:
            vals = [cache.get(f"tau{tau:g}_on{on:g}_off{off:g}_seed{s}") for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            for i, kind in enumerate(("tend", "win")):
                a = np.array([v[i] for v in vals])
                sd = a.std(ddof=1) if len(a) > 1 else 0.0
                agg[(kind, tau, on, off)] = (a.mean(), sd/np.sqrt(len(a)) if ERRBAR == "sem" else sd)
            a_t = np.array([v[0] for v in vals]); a_w = np.array([v[1] for v in vals])
            rows.append(dict(tau_m_s=tau, on_s=on, off_s=off, n_seeds=len(vals),
                             duty=round(on/(on+off), 5),
                             tend_mean=round(a_t.mean(), 4),
                             tend_std=round(a_t.std(ddof=1) if len(a_t) > 1 else 0, 4),
                             win_mean=round(a_w.mean(), 4),
                             win_std=round(a_w.std(ddof=1) if len(a_w) > 1 else 0, 4),
                             win_sem=round((a_w.std(ddof=1)/np.sqrt(len(a_w))) if len(a_w) > 1 else 0, 4)))
with open(FIGS/"pulse_ratio.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

_v = [v for (k, *_), (v, e) in agg.items()]
YLIM = (min(0.9, min(_v)*0.9), max(_v)*1.6)

ELAB = {"sem": "standard error of the seed mean", "std": "seed-to-seed spread (5 seeds)"}
YLAB = {"tend": r"$\rho_{\rm in}\,/\,\rho_{\rm out}$  ($t_{\rm end}$)",
        "win":  r"$\rho_{\rm in}\,/\,\rho_{\rm out}$  (mean over last 419 s)"}

plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.0,
                     "xtick.direction": "in", "ytick.direction": "in"})


def draw(ax, tau, kind):
    ax.axhline(1.0, color="#9a9a95", lw=1.0, ls=(0, (4, 3)), zorder=1)
    for on, col in zip(ONS, COLORS):
        pts = [(o,) + agg[(kind, tau, on, o)] for o in OFFS if (kind, tau, on, o) in agg]
        if not pts:
            continue
        o, v, e = zip(*pts)
        ax.errorbar(o, v, yerr=e, color=col, lw=2.0, marker="o", ms=7.5,
                    mec="white", mew=1.2, capsize=0, elinewidth=2.0,
                    label=f"{on} s ON", zorder=3)
    ax.set(xscale="log", yscale="log", ylim=YLIM)
    ax.set_xticks(OFFS); ax.set_xticklabels([f"{o:g}" for o in OFFS], fontsize=9)
    ax.minorticks_off()
    ax.set_title(rf"$\tau_m$ = {tau:g} s")
    ax.grid(True, which="major", color="#e6e6e2", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


for kind in ("tend", "win"):
    nr = 2 if len(TAUS) > 2 else 1
    nc = -(-len(TAUS)//nr)
    fig, axes = plt.subplots(nr, nc, figsize=(6.4*nc, 4.7*nr), sharey=True, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, tau in zip(axes, TAUS):
        draw(ax, tau, kind)
    for ax in axes[len(TAUS):]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[:len(TAUS)]):
        if i % nc == 0: ax.set_ylabel(YLAB[kind])
        if i >= len(TAUS) - nc: ax.set_xlabel("dark duration (s)")
    # legenda costruita a parte: non dipende da quale pannello ha dati
    handles = [plt.Line2D([], [], color=c, lw=2.0, marker="o", ms=7.5, mec="white",
                          mew=1.2, label=f"{on} s ON") for on, c in zip(ONS, COLORS)]
    fig.legend(handles=handles, frameon=False, ncol=len(ONS), fontsize=11,
               loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"Accumulation inside the ring (r < {r_ring:.0f} µm) under pulsed light, 180 µW ring "
                 f"— {len(SEEDS)} seeds per point", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.text(0.01, 0.01, f"dashed line: 1 = uniform density    |    bars: {ELAB[ERRBAR]}",
             ha="left", va="bottom", color="#52514e", fontsize=10)
    FIGS_EN.mkdir(exist_ok=True)
    fig.savefig(FIGS_EN/f"pulse_ratio_{kind}_both_tau.png", dpi=200)

    for tau in TAUS:
        f1, a = plt.subplots(figsize=(6.8, 5.0))
        draw(a, tau, kind)
        a.set(ylabel=YLAB[kind], xlabel="dark duration (s)")
        a.legend(handles=[plt.Line2D([], [], color=c, lw=2.0, marker="o", ms=7.5,
                                     mec="white", mew=1.2, label=f"{on} s ON")
                          for on, c in zip(ONS, COLORS)],
                 frameon=False, loc="lower left", fontsize=10)
        f1.tight_layout(); f1.savefig(FIGS_EN/f"pulse_ratio_{kind}_tau{tau:g}.png", dpi=200)

print(f"figures in {FIGS_EN}, csv in {FIGS}")
