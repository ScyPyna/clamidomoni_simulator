"""Figures and summary for theta_velocity.py (English labels, written to figs/en/).

  theta_vel_method.png   P(theta) inside the ring: lab method (velocity over 0.51 s) vs instantaneous orientation
  theta_vel_states.png   lab method, split by reversal state
  theta_vel_window.png   <cos theta> of each state vs the velocity window
  theta_vel_summary.csv  per condition: reversed fraction, <cos theta> and U depth, both methods
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_lang  # noqa: F401  Italian labels when FIG_LANG=it

FIGS  = Path("/media/user/LaCie/pulsed_exps/figs")
CACHE = FIGS/"theta_vel_cache"
EN    = FIGS/"en"
EDGES = np.linspace(0, np.pi, 11)
CEN   = 0.5*(EDGES[1:] + EDGES[:-1])
WIDTH = EDGES[1] - EDGES[0]
DT    = 0.1024
WINDOWS = [1, 2, 5, 10]
THRS  = [("3.07084", "3.0708"), ("2.6", "2.6")]
TAUS, ONS, OFFS, SEEDS = [8, 16], [10, 15], [1, 5], range(5)
MIN_SHARE = 0.03      # a state is shown only if it holds at least 3 % of the samples
C_NORM, C_REV, C_TOT, C_INST = "#2a78d6", "#eb6834", "#2b2b2b", "#9a9a95"
MUTED, GRID = "#6b6b66", "#e6e6e2"
plt.rcParams.update({"font.size": 10.5, "axes.linewidth": 1.0, "axes.edgecolor": "#55554f",
                     "xtick.direction": "in", "ytick.direction": "in"})


def load(thr, tau, on, off):
    return [np.load(CACHE/f"hires_thr{thr}_tau{tau}_on{on}_off{off}_seed{s}.npz") for s in SEEDS]


def get(zs, method, region, state, light="all"):
    """Pooled density histogram, per-seed <cos>, and sample count."""
    k = f"{method}|{region}|{state}|{light}"
    h = sum(z[k+"|h"] for z in zs).astype(float)
    n = sum(int(z[k+"|n"]) for z in zs)
    per_seed = np.array([float(z[k+"|s"])/int(z[k+"|n"]) for z in zs if int(z[k+"|n"]) > 0])
    dens = h/(h.sum()*WIDTH) if h.sum() > 0 else np.full(len(CEN), np.nan)
    return dens, per_seed, n


def u_depth(p):
    """Mean of the two end bins minus mean of the two central bins (P units)."""
    return 0.5*(p[0] + p[-1]) - 0.5*(p[4] + p[5])


conds = [(thr, lab, tau, on, off) for thr, lab in THRS for tau in TAUS for on in ONS for off in OFFS]
rows = []
for thr, lab, tau, on, off in conds:
    zs = load(thr, tau, on, off)
    row = dict(threshold=lab, tau_m_s=tau, on_s=on, off_s=off)
    _, _, n_all = get(zs, "w5", "in", "all"); _, _, n_rev = get(zs, "w5", "in", "rev")
    row["reversed_in_pct"] = round(100*n_rev/n_all, 1)
    for m, tag in (("inst", "inst"), ("w5", "lab")):
        for st in ("all", "norm", "rev"):
            _, c, _ = get(zs, m, "in", st)
            row[f"cos_{tag}_{st}"] = round(c.mean(), 4) if len(c) else ""
        p, _, _ = get(zs, m, "in", "all")
        row[f"U_{tag}"] = round(u_depth(p), 4)
    _, c, _ = get(zs, "w5", "out", "all"); row["cos_lab_all_outside"] = round(c.mean(), 4)
    for lt in ("on", "off"):
        _, c, _ = get(zs, "w5", "in", "norm", lt); row[f"cos_lab_norm_light{lt}"] = round(c.mean(), 4) if len(c) else ""
    rows.append(row)
with open(FIGS/"theta_vel_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def grid(kind, fname, title):
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 11), sharex=True, sharey=True)
    for i, (thr, lab) in enumerate(THRS):
        for j, tau in enumerate(TAUS):
            r = 2*i + j
            for c, (on, off) in enumerate([(on, off) for on in ONS for off in OFFS]):
                ax = axes[r, c]
                zs = load(thr, tau, on, off)
                ax.axhline(1/np.pi, color=C_INST, lw=0.9, ls=(0, (2, 3)))
                if kind == "method":
                    p, _, _ = get(zs, "inst", "in", "all")
                    ax.plot(CEN, p, color=C_INST, lw=2.0, ls=(0, (4, 2)))
                    p, _, _ = get(zs, "w5", "in", "all")
                    ax.plot(CEN, p, color=C_TOT, lw=2.0, marker="o", ms=3.5)
                    ax.text(0.5, 0.95, f"U depth {u_depth(p):.3f}", transform=ax.transAxes,
                            ha="center", va="top", fontsize=8.5, color=MUTED)
                else:
                    _, _, ntot = get(zs, "w5", "in", "all")
                    for st, col in (("norm", C_NORM), ("rev", C_REV)):
                        p, _, n = get(zs, "w5", "in", st)
                        if n >= MIN_SHARE*ntot:
                            ax.plot(CEN, p, color=col, lw=2.0)
                    p, _, _ = get(zs, "w5", "in", "all")
                    ax.plot(CEN, p, color=C_TOT, lw=1.6, ls=(0, (4, 2)))
                    _, _, na = get(zs, "w5", "in", "all"); _, _, nr = get(zs, "w5", "in", "rev")
                    ax.text(0.97, 0.95, f"{100*nr/na:.0f}% reversed", transform=ax.transAxes,
                            ha="right", va="top", fontsize=8.5, color=MUTED)
                ax.set(xlim=(0, np.pi))
                ax.set_xticks([0, np.pi/2, np.pi]); ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
                ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
                if r == 0:
                    ax.set_title(f"light {on} s / dark {off} s", fontsize=10.5)
                if c == 0:
                    ax.set_ylabel(f"threshold {lab}\n" + rf"$\tau_m$ = {tau} s", fontsize=10)
    if kind == "method":
        axes[0, 0].set_ylim(0.2, 0.45)
        handles = [plt.Line2D([], [], color=C_TOT, lw=2, marker="o", ms=4,
                              label="lab method: velocity over 0.51 s"),
                   plt.Line2D([], [], color=C_INST, lw=2, ls=(0, (4, 2)),
                              label="instantaneous swimming direction")]
    else:
        axes[0, 0].set_ylim(0.15, 0.55)
        handles = [plt.Line2D([], [], color=C_NORM, lw=2, label="not reversed"),
                   plt.Line2D([], [], color=C_REV, lw=2, label="reversed"),
                   plt.Line2D([], [], color=C_TOT, lw=1.6, ls=(0, (4, 2)), label="total")]
    fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="upper center",
               bbox_to_anchor=(0.5, 0.965), fontsize=10.5)
    fig.suptitle(title, y=0.99, fontsize=13)
    fig.supxlabel(r"$\theta$ (angle between motion and light gradient)", fontsize=11)
    fig.supylabel(r"$P(\theta)$ inside the ring (r ≤ 300 µm)", fontsize=11)
    fig.text(0.01, 0.005, "5 seeds pooled, last 1258 s of each run, 10 bins as in the lab pipeline. "
             "Dotted line: isotropic, 1/π. A state holding < 3 % of samples is not drawn.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0.01, 0.015, 1, 0.945))
    fig.savefig(EN/fname, dpi=180)
    plt.close(fig)


grid("method", "theta_vel_method.png",
     "P(θ) computed as in the lab pipeline vs from the instantaneous direction")
grid("states", "theta_vel_states.png",
     "P(θ) from 0.51 s velocities, split by the reversal state")

# ------------------------------------------------------------------ window dependence
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
xs = [0] + [w*DT for w in WINDOWS]
for ax, (thr, lab) in zip(axes, THRS):
    ax.axhline(0, color=C_INST, lw=1.0, ls=(0, (4, 3)))
    ax.axvline(5*DT, color="#d9d9d4", lw=6, zorder=0)
    for st, col in (("norm", C_NORM), ("rev", C_REV), ("all", C_TOT)):
        allv = []
        for _, _, tau, on, off in [c for c in conds if c[0] == thr]:
            zs = load(thr, tau, on, off)
            if get(zs, "w5", "in", st)[2] < MIN_SHARE*get(zs, "w5", "in", "all")[2]:
                continue
            v = [get(zs, "inst", "in", st)[1].mean()] + [get(zs, f"w{w}", "in", st)[1].mean() for w in WINDOWS]
            allv.append(v)
            ax.plot(xs, v, color=col, lw=0.8, alpha=0.35)
        ax.plot(xs, np.nanmedian(allv, axis=0), color=col, lw=2.6, marker="o", ms=6, mec="white",
                label={"norm": "not reversed", "rev": "reversed", "all": "total"}[st])
    ax.set(xlabel="velocity window (s); 0 = instantaneous direction", title=f"threshold {lab}")
    ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.text(5*DT, 0.02, " lab\n window", transform=ax.get_xaxis_transform(), fontsize=8.5, color=MUTED)
axes[0].set_ylabel(r"$\langle\cos\theta\rangle$ inside the ring")
fig.legend(handles=axes[0].get_legend_handles_labels()[0], labels=axes[0].get_legend_handles_labels()[1],
           frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.93), fontsize=10)
fig.suptitle("Averaging the velocity over longer windows sharpens both states, the total stays at zero",
             fontsize=12.5, y=0.99)
fig.text(0.01, 0.01, "Thin lines: the 8 conditions (τm 8/16 s, light 10/15 s, dark 1/5 s); thick: median. "
         "Positive = moving towards the light. States holding < 3 % of samples are left out.", color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 0.88))
fig.savefig(EN/"theta_vel_window.png", dpi=190)
plt.close(fig)

for r in rows:
    print("thr %-6s tau %2d on %2d off %d  rev %5.1f%%  lab: all %+.3f norm %+.3f rev %+.3f U %.3f | inst: all %+.3f norm %+.3f rev %+.3f U %.3f | out %+.3f | norm on/off %s/%s" % (
        r["threshold"], r["tau_m_s"], r["on_s"], r["off_s"], r["reversed_in_pct"],
        r["cos_lab_all"], r["cos_lab_norm"], r["cos_lab_rev"] or 0, r["U_lab"],
        r["cos_inst_all"], r["cos_inst_norm"], r["cos_inst_rev"] or 0, r["U_inst"],
        r["cos_lab_all_outside"], r["cos_lab_norm_lighton"], r["cos_lab_norm_lightoff"]))
