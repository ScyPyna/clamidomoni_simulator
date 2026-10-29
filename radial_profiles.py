"""Radial density profiles rho(r): simulations against the lab (440 nm, 180 uW ring).

Lab data: lab_data/radial_density_lastlight_440nm.json, extracted from the vector paths of
radial_density_lastlight.pdf (one frame at the end of the last light phase, 50 um bins).
The lab absolute density changes a lot between acquisitions (number of cells in the field), so
every profile, lab or simulated, is divided by its area-weighted mean over r < 600 um and only
the shape is compared.

Simulations: 5 seeds pooled, light-on frames of the last 419 s (the lab frame is taken with the
light on), same 50 um bins. Per-run profiles are cached in figs/radial_cache.csv.

Outputs (figs/en/, or figs/it/ with FIG_LANG=it):
  radial_grid.png       lab vs simulated shapes, one panel per light/dark protocol
  radial_contrast.png   centre / ring contrast, lab vs simulations at every threshold
and figs/radial_summary.csv with the contrast and the shape misfit of every model variant.
"""
import csv, glob, json, os, subprocess
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_lang  # noqa: F401  Italian labels when FIG_LANG=it

OUT   = Path("/media/user/LaCie/pulsed_exps")
FIGS  = OUT/"figs"
EN    = FIGS/"en"
CACHE = FIGS/"radial_cache.csv"
LAB   = Path(__file__).resolve().parent/"lab_data"/"radial_density_lastlight_440nm.json"
N, WINDOW = 371, 512
EDGES = np.arange(0, 601, 50.0)
CEN   = 0.5*(EDGES[1:] + EDGES[:-1])
AREA  = np.pi*(EDGES[1:]**2 - EDGES[:-1]**2)
ONS   = [1, 5, 10, 15]
OFFS  = [0.01, 0.1, 1, 5, 10]
TAUS  = [4, 8, 16, 20]
THRS  = ["2.2", "2.4", "2.6", "2.8", "3", "3.0708", "3.2"]
TAU_COL = {4: "#9fd4b4", 8: "#55a97f", 16: "#23734f", 20: "#0b412b"}
INK, MUTED, GRID = "#2b2b2b", "#6b6b66", "#e6e6e2"


def run_name(th, tau, on, off, s):
    return (f"tau{tau}_on{on}_off{off:g}_seed{s}" if th == "3.0708"
            else f"thr{th}_tau{tau}_on{on}_off{off:g}_seed{s}")


def tail_frames(path):
    """Last WINDOW frames of a log, read in one sequential block.

    `tail -n` walks the file backwards in small reads, which is very slow on the exFAT drive.
    Here the byte count is estimated from the line length, with a margin, and trimmed after.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - 200_000))
        sample = f.read().splitlines()[1:-1]
        per_line = max(len(x) for x in sample) + 1
        need = int(per_line*N*WINDOW*1.05)
        f.seek(max(0, size - need))
        lines = f.read().decode().splitlines()[1:]          # first line may be cut
    lines = lines[-N*WINDOW:]
    lines = lines[len(lines) % N:]
    d = np.loadtxt(lines)
    # frames must be whole: every block of N lines starts with particle 0
    assert d[0, 1] == 0, path
    return d


def profile(run):
    """Counts per radial bin over light-on frames of the last WINDOW frames, and frame count."""
    logs = glob.glob(str(OUT/run/"logs"/"Simlog*"))
    if not logs:
        return run, None
    d = tail_frames(logs[0])
    nf = len(d)//N
    d = d[:nf*N].reshape(nf, N, -1)
    on = d[:, 0, 7] > 0.5
    if not on.any():
        return run, None
    r = np.hypot(d[on, :, 2], d[on, :, 3]).ravel()*10
    h, _ = np.histogram(r, bins=EDGES)
    return run, (h, int(on.sum()))


def shape(counts):
    """Density per bin divided by the area-weighted mean density over r < 600 um."""
    rho = counts/AREA
    return rho/(counts.sum()/AREA.sum())


def lab_shapes():
    raw = json.load(open(LAB))
    out = {}
    for on, series in raw.items():
        for off, pts in series.items():
            rho = np.full(len(CEN), np.nan)
            for r, v in pts:
                rho[np.argmin(abs(CEN - r))] = v
            ok = np.isfinite(rho)
            mean = np.nansum(rho*AREA)/AREA[ok].sum()
            out[(int(on), float(off))] = rho/mean
    return out


def contrast(p):
    """Centre (0-100 um) over ring (150-250 um) density."""
    return np.nanmean(p[0:2])/np.nanmean(p[3:5])


if __name__ == "__main__":
    runs = sorted({run_name(th, tau, on, off, s) for th in THRS for tau in TAUS for on in ONS
                   for off in OFFS for s in range(5)})
    runs = [r for r in runs if (OUT/r).exists()]
    cache = {}
    if CACHE.exists():
        for row in csv.reader(open(CACHE)):
            if row[0] != "run":
                cache[row[0]] = (np.array(row[2:], dtype=float), int(row[1]))
    todo = [r for r in runs if r not in cache]

    def save():
        with open(CACHE, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["run", "frames"] + [f"{a:.0f}-{b:.0f}" for a, b in zip(EDGES[:-1], EDGES[1:])])
            for k, (h, n) in sorted(cache.items()):
                w.writerow([k, n] + list(np.asarray(h).astype(int)))

    # few workers: the logs sit on a spinning disk, and many parallel readers make it seek-bound
    with Pool(3) as pool:
        for i, (run, v) in enumerate(pool.imap_unordered(profile, todo), 1):
            if v is not None:
                cache[run] = v
            if i % 50 == 0:
                save(); print(f"  {i}/{len(todo)}", flush=True)
    save()
    print(f"{len(cache)} runs in cache ({len(todo)} computed now)")

    def sim(th, tau, on, off):
        hs = [cache.get(run_name(th, tau, on, off, s)) for s in range(5)]
        hs = [h for h, _ in (x for x in hs if x)]
        return shape(np.sum(hs, axis=0)) if len(hs) >= 3 else None

    lab = lab_shapes()

    # ------------------------------------------------------------- summary table
    rows = []
    for (on, off), lp in sorted(lab.items()):
        row = dict(on_s=on, off_s=off, lab_contrast=round(contrast(lp), 2))
        for th in THRS:
            for tau in TAUS:
                p = sim(th, tau, on, off)
                if p is None:
                    continue
                ok = np.isfinite(lp) & (p > 0) & (lp > 0)
                misfit = np.sqrt(np.mean((np.log10(p[ok]) - np.log10(lp[ok]))**2))
                row[f"thr{th}_tau{tau}_contrast"] = round(contrast(p), 2)
                row[f"thr{th}_tau{tau}_misfit"] = round(misfit, 3)
        rows.append(row)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("on_s", "off_s", "lab_contrast"), k))
    with open(FIGS/"radial_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    plt.rcParams.update({"font.size": 10.5, "axes.linewidth": 1.0, "axes.edgecolor": "#55554f",
                         "xtick.direction": "in", "ytick.direction": "in"})

    # ------------------------------------------------------------- grid of shapes
    fig, axes = plt.subplots(len(ONS), len(OFFS), figsize=(15, 11.5), sharex=True, sharey=True)
    for i, on in enumerate(ONS):
        for j, off in enumerate(OFFS):
            ax = axes[i, j]
            ax.axvspan(175, 265, color="#ededea", zorder=0)
            ax.axhline(1, color="#9a9a95", lw=0.9, ls=(0, (4, 3)))
            for tau in TAUS:
                p = sim("3.0708", tau, on, off)
                if p is not None:
                    ax.plot(CEN, p, color=TAU_COL[tau], lw=1.9, zorder=2)
            lp = lab.get((on, off))
            if lp is not None:
                ax.plot(CEN, lp, ls="none", marker="^", ms=6.5, color=INK, mec="white", mew=0.8, zorder=3)
                ax.text(0.97, 0.95, f"lab C = {contrast(lp):.1f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8.5, color=INK)
            else:
                ax.text(0.5, 0.2, "no lab data", transform=ax.transAxes, ha="center", color=MUTED, fontsize=9)
            ax.set(yscale="log", ylim=(0.03, 30), xlim=(0, 600))
            ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f"dark {off:g} s", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"light on {on} s", fontsize=11)
    handles = [plt.Line2D([], [], ls="none", marker="^", ms=7, color=INK, label="lab (440 nm, end of last light phase)")]
    handles += [plt.Line2D([], [], color=TAU_COL[t], lw=2, label=rf"sim, $\tau_m$ = {t} s") for t in TAUS]
    fig.legend(handles=handles, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.965), fontsize=10)
    fig.suptitle("Radial density shape: the lab accumulates in the centre of the ring", y=0.99, fontsize=13)
    fig.supxlabel("distance from the ring centre r (µm)", fontsize=11)
    fig.supylabel("ρ(r) / mean density within 600 µm", fontsize=11)
    fig.text(0.01, 0.005, "Grey band: the illuminated ring (light above half its peak). C = centre (0-100 µm) / ring (150-250 µm). "
             "Simulations at the working threshold 3.0708, 5 seeds, light-on frames of the last 419 s.",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0.01, 0.015, 1, 0.945))
    EN.mkdir(exist_ok=True)
    fig.savefig(EN/"radial_grid.png", dpi=170)
    plt.close(fig)

    # ------------------------------------------------------------- centre / ring contrast
    fig, axes = plt.subplots(1, len(ONS), figsize=(15, 4.6), sharey=True)
    for ax, on in zip(axes, ONS):
        ax.axhline(1, color="#9a9a95", lw=0.9, ls=(0, (4, 3)))
        for tau in TAUS:
            ys = [sim("3.0708", tau, on, off) for off in OFFS]
            ax.plot(OFFS, [contrast(p) if p is not None else np.nan for p in ys], color=TAU_COL[tau], lw=2,
                    marker="o", ms=5.5, mec="white", label=rf"sim, $\tau_m$ = {tau} s")
        lo = [(off, contrast(lab[(on, off)])) for off in OFFS if (on, off) in lab]
        ax.plot(*zip(*lo), ls="none", marker="^", ms=9, color=INK, mec="white", mew=1, zorder=4, label="lab")
        ax.set(xscale="log", yscale="log", title=f"light on {on} s", xlabel="dark duration (s)")
        ax.set_xticks(OFFS); ax.set_xticklabels([f"{o:g}" for o in OFFS])
        ax.minorticks_off(); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    axes[0].set_ylabel("centre / ring density  (0-100 µm / 150-250 µm)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Above 1: more cells in the centre than on the ring", y=0.99, fontsize=12.5)
    fig.text(0.01, 0.01, "Simulations at the working threshold 3.0708. Dashed line: centre as dense as the ring.",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(EN/"radial_contrast.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------- console summary
    print("\ncentre/ring contrast C and shape misfit (rms of log10 ratio; lower = better)")
    for r in rows:
        best = sorted(((r[k], k[:-7]) for k in r if k.endswith("_misfit")))[:3]
        print("ON %2d OFF %-5g lab C %5.2f | working thr C: %s | best: %s" % (
            r["on_s"], r["off_s"], r["lab_contrast"],
            " ".join(f"τ{t}={r.get(f'thr3.0708_tau{t}_contrast', float('nan')):.2f}" for t in TAUS),
            ", ".join(f"{k} ({v:.2f})" for v, k in best)))
    # overall ranking of model variants over the conditions where all exist
    variants = sorted({k[:-7] for r in rows for k in r if k.endswith("_misfit")})
    print("\nmean misfit per variant (conditions available):")
    for v in sorted(variants, key=lambda v: np.mean([r[v+"_misfit"] for r in rows if v+"_misfit" in r])):
        m = [r[v+"_misfit"] for r in rows if v+"_misfit" in r]
        print(f"  {v:18s} {np.mean(m):.3f}  (n={len(m)})")
