"""Direct comparison of the lab radial profiles with simulations run on the lab's own maps (batch 6).

Three questions, all on the shape of rho(r) (normalised as in radial_profiles.py):
  1. per condition, which tau_m in {4, 8, 16} s reproduces the lab profile best;
  2. consistency: does a single tau_m explain all 18 conditions? Leave-one-condition-out:
     tau chosen on 17 conditions, the 18th predicted, compared with its own best tau;
  3. geometry bias: simulations on the lab maps at a known tau are treated as data and
     matched against simulations on the old ring_180muW map (main grid). The tau that comes
     out is what an analysis with the wrong map would report.

Outputs (figs/en/, or figs/it/ with FIG_LANG=it): labmap_radial_grid.png, labmap_tau_misfit.png;
tables figs/labmap_fit.csv and figs/labmap_geometry_bias.csv.
"""
import csv
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_lang  # noqa: F401  Italian labels when FIG_LANG=it
import radial_profiles as rp

TAUS = [4, 8, 16]
CONDS = [(98, 1, 0.01), (99, 1, 0.1), (99, 1, 1), (99, 5, 0.01), (100, 5, 0.1), (100, 5, 1),
         (101, 5, 5), (102, 5, 10), (103, 10, 0.01), (104, 10, 0.1), (104, 10, 1), (104, 10, 5),
         (104, 10, 10), (105, 15, 0.01), (105, 15, 0.1), (105, 15, 1), (105, 15, 5), (105, 15, 10)]
TAU_COL = rp.TAU_COL
INK, MUTED, GRID = rp.INK, rp.MUTED, rp.GRID


def lab_run(sess, tau, on, off, s):
    return f"lab{sess}_tau{tau}_on{on}_off{off:g}_seed{s}"


def old_run(tau, on, off, s):
    return f"tau{tau}_on{on}_off{off:g}_seed{s}"


def misfit(p, q):
    ok = np.isfinite(p) & np.isfinite(q) & (p > 0) & (q > 0)
    return np.sqrt(np.mean((np.log10(p[ok]) - np.log10(q[ok]))**2))


if __name__ == "__main__":
    # ---------------------------------------------------------------- profiles (shared cache)
    cache = {}
    if rp.CACHE.exists():
        for row in csv.reader(open(rp.CACHE)):
            if row[0] != "run":
                cache[row[0]] = (np.array(row[2:], dtype=float), int(row[1]))
    want = [lab_run(se, t, on, off, s) for se, on, off in CONDS for t in TAUS for s in range(5)]
    want += [old_run(t, on, off, s) for _, on, off in CONDS for t in TAUS for s in range(5)]
    todo = [r for r in want if r not in cache and (rp.OUT/r).exists()]
    with Pool(3) as pool:
        for run, v in pool.imap_unordered(rp.profile, todo):
            if v is not None:
                cache[run] = v
    with open(rp.CACHE, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "frames"] + [f"{a:.0f}-{b:.0f}" for a, b in zip(rp.EDGES[:-1], rp.EDGES[1:])])
        for k, (h, n) in sorted(cache.items()):
            w.writerow([k, n] + list(np.asarray(h).astype(int)))
    print(f"{len(todo)} profiles computed now")

    def shape_of(names):
        hs = [cache[n][0] for n in names if n in cache]
        return rp.shape(np.sum(hs, axis=0)) if len(hs) >= 3 else None

    def seeds_of(names):
        return [rp.shape(cache[n][0]) for n in names if n in cache]

    lab = rp.lab_shapes()
    sim_new = {(se, on, off, t): shape_of([lab_run(se, t, on, off, s) for s in range(5)])
               for se, on, off in CONDS for t in TAUS}
    sim_old = {(on, off, t): shape_of([old_run(t, on, off, s) for s in range(5)])
               for _, on, off in CONDS for t in TAUS}

    # ---------------------------------------------------------------- 1. per condition
    M = np.full((len(CONDS), len(TAUS)), np.nan)
    M_old = np.full_like(M, np.nan)
    rows = []
    for i, (se, on, off) in enumerate(CONDS):
        lp = lab[(on, off)]
        for j, t in enumerate(TAUS):
            M[i, j] = misfit(sim_new[(se, on, off, t)], lp)
            M_old[i, j] = misfit(sim_old[(on, off, t)], lp)
        rows.append(dict(session=se, on_s=on, off_s=off, lab_C=round(rp.contrast(lp), 2),
                         **{f"C_labmap_tau{t}": round(rp.contrast(sim_new[(se, on, off, t)]), 2) for t in TAUS},
                         **{f"misfit_labmap_tau{t}": round(M[i, j], 3) for j, t in enumerate(TAUS)},
                         **{f"misfit_oldmap_tau{t}": round(M_old[i, j], 3) for j, t in enumerate(TAUS)},
                         best_tau_labmap=TAUS[int(np.nanargmin(M[i]))],
                         best_tau_oldmap=TAUS[int(np.nanargmin(M_old[i]))]))
    with open(rp.FIGS/"labmap_fit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("\ncondition            lab C | C with lab maps tau 4/8/16      | misfit lab maps 4/8/16  | old map 4/8/16        | best lab/old")
    for r in rows:
        print(f"S{r['session']} ON{r['on_s']:>2} OFF{r['off_s']:<5g} {r['lab_C']:6.2f} | "
              + " ".join(f"{r[f'C_labmap_tau{t}']:6.2f}" for t in TAUS) + "            | "
              + " ".join(f"{r[f'misfit_labmap_tau{t}']:.2f}" for t in TAUS) + "      | "
              + " ".join(f"{r[f'misfit_oldmap_tau{t}']:.2f}" for t in TAUS) + f"      | {r['best_tau_labmap']:>2}/{r['best_tau_oldmap']:>2}")
    print("\nmean misfit over the 18 conditions   lab maps: " + "  ".join(f"tau{t} {np.nanmean(M[:, j]):.3f}" for j, t in enumerate(TAUS))
          + "   | old map: " + "  ".join(f"tau{t} {np.nanmean(M_old[:, j]):.3f}" for j, t in enumerate(TAUS)))
    best = TAUS[int(np.nanargmin(np.nanmean(M, axis=0)))]
    wins = {t: int(sum(TAUS[int(np.nanargmin(M[i]))] == t for i in range(len(CONDS)))) for t in TAUS}
    print(f"best single tau (lab maps): {best} s; conditions won by each tau: {wins}")

    # ---------------------------------------------------------------- 2. leave one condition out
    hits, excess = 0, []
    for i in range(len(CONDS)):
        keep = [k for k in range(len(CONDS)) if k != i]
        t_sel = int(np.nanargmin(np.nanmean(M[keep], axis=0)))
        hits += (t_sel == int(np.nanargmin(M[i])))
        excess.append(M[i, t_sel] - np.nanmin(M[i]))
    print(f"leave-one-out: tau chosen on the other 17 is the held-out condition's own best in {hits}/18; "
          f"mean extra misfit {np.mean(excess):.3f}")
    # seed-to-seed noise floor of the misfit, for scale
    floor = []
    for se, on, off in CONDS:
        for t in TAUS:
            ss = seeds_of([lab_run(se, t, on, off, s) for s in range(5)])
            m = sim_new[(se, on, off, t)]
            floor += [misfit(x, m) for x in ss]
    print(f"noise floor: misfit of a single seed against its own 5-seed mean = {np.median(floor):.3f} (median)")

    # ---------------------------------------------------------------- 3. geometry bias
    brows = []
    print("\ngeometry bias: lab-map simulations at a known tau, matched against the old map")
    for t_true in TAUS:
        picks = []
        for se, on, off in CONDS:
            data = sim_new[(se, on, off, t_true)]
            mm = [misfit(sim_old[(on, off, t)], data) for t in TAUS]
            picks.append(TAUS[int(np.nanargmin(mm))])
        pooled = [np.nanmean([misfit(sim_old[(on, off, t)], sim_new[(se, on, off, t_true)]) for se, on, off in CONDS]) for t in TAUS]
        t_pooled = TAUS[int(np.argmin(pooled))]
        counts = {t: picks.count(t) for t in TAUS}
        brows.append(dict(tau_true=t_true, tau_recovered_all_conditions=t_pooled,
                          **{f"conditions_picking_tau{t}": counts[t] for t in TAUS},
                          **{f"mean_misfit_tau{t}": round(p, 3) for t, p in zip(TAUS, pooled)}))
        print(f"  true tau {t_true:2d} s -> recovered {t_pooled:2d} s (per condition: {counts}); "
              f"mean misfit " + " ".join(f"{p:.2f}" for p in pooled))
    with open(rp.FIGS/"labmap_geometry_bias.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0])); w.writeheader(); w.writerows(brows)

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 10.5, "axes.linewidth": 1.0, "axes.edgecolor": "#55554f",
                         "xtick.direction": "in", "ytick.direction": "in"})
    ONS, OFFS = [1, 5, 10, 15], [0.01, 0.1, 1, 5, 10]
    fig, axes = plt.subplots(4, 5, figsize=(15, 11.5), sharex=True, sharey=True)
    for i, on in enumerate(ONS):
        for j, off in enumerate(OFFS):
            ax = axes[i, j]
            ax.axhline(1, color="#9a9a95", lw=0.9, ls=(0, (4, 3)))
            cond = [c for c in CONDS if c[1] == on and c[2] == off]
            if cond:
                se = cond[0][0]
                for t in TAUS:
                    p = sim_new[(se, on, off, t)]
                    if p is not None:
                        ax.plot(rp.CEN, p, color=TAU_COL[t], lw=1.9)
                lp = lab[(on, off)]
                ax.plot(rp.CEN, lp, ls="none", marker="^", ms=6.5, color=INK, mec="white", mew=0.8, zorder=3)
                ax.text(0.97, 0.95, f"session {se}", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)
            else:
                ax.text(0.5, 0.2, "no lab data", transform=ax.transAxes, ha="center", color=MUTED, fontsize=9)
            ax.set(yscale="log", ylim=(0.03, 30), xlim=(0, 600))
            ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f"dark {off:g} s", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"light on {on} s", fontsize=11)
    handles = [plt.Line2D([], [], ls="none", marker="^", ms=7, color=INK, label="lab (440 nm, end of last light phase)")]
    handles += [plt.Line2D([], [], color=TAU_COL[t], lw=2, label=rf"sim on the lab map, $\tau_m$ = {t} s") for t in TAUS]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.965), fontsize=10)
    fig.suptitle("Radial density shape with the lab's own illumination maps", y=0.99, fontsize=13)
    fig.supxlabel("distance from the ring centre r (µm)", fontsize=11)
    fig.supylabel("ρ(r) / mean density within 600 µm", fontsize=11)
    fig.text(0.01, 0.005, "Each condition simulated with the map and power of its acquisition session, "
             "working threshold 3.0708, 5 seeds.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0.01, 0.015, 1, 0.945))
    rp.EN.mkdir(exist_ok=True)
    fig.savefig(rp.EN/"labmap_radial_grid.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    labels = [f"S{se}  {on}/{off:g}" for se, on, off in CONDS]
    vmax = np.nanmax([M, M_old])
    for ax, A, title in ((axes[0], M, "lab maps"), (axes[1], M_old, "old ring_180muW map")):
        im = ax.imshow(A, cmap="Greens_r", vmin=0, vmax=vmax, aspect="auto")
        for i in range(A.shape[0]):
            k = int(np.nanargmin(A[i]))
            for j in range(A.shape[1]):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        fontweight="bold" if j == k else "normal",
                        color="white" if A[i, j] < 0.45*vmax else INK)
        ax.set_xticks(range(len(TAUS))); ax.set_xticklabels([f"{t} s" for t in TAUS])
        ax.set(xlabel=r"$\tau_m$", title=f"shape misfit, simulations on the {title}")
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(CONDS))); axes[0].set_yticklabels(labels, fontsize=8.5)
    axes[0].set_ylabel("session, light / dark (s)")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02); cb.set_label("rms of log10(sim / lab)")
    fig.suptitle("Which memory reproduces each lab profile (bold: best per condition)", fontsize=12.5)
    fig.savefig(rp.EN/"labmap_tau_misfit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
