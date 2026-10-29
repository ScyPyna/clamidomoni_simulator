"""Alternative views of the same grid, built to show one thing: rho encodes the memory
time only when the light phase lasts at least as long as the memory.

Figure A: rho against tau_m, one line per ON time, one panel per dark duration. A flat
line means the density cannot tell the memories apart; a sloped one means it can.

Figure B: the same information compressed to a sensitivity, d log(rho) / d log(tau_m),
over the whole protocol grid. Left panel fits the slope across the full memory range
(4 -> 20 s), right panel only across the closest pair (16 -> 20 s), which is the one that
actually matters when identifying a long memory.

Reads the aggregated CSV written by pulse_ratio.py.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_lang  # noqa: F401  Italian labels when FIG_LANG=it
from matplotlib.colors import LinearSegmentedColormap

FIGS  = "/media/user/LaCie/pulsed_exps/figs"
TAUS  = [4.0, 8.0, 16.0, 20.0]
ONS   = [1, 5, 10, 15, 20]
OFFS  = [0.01, 0.1, 0.5, 1, 2, 5, 7, 10, 15, 20]
PANEL_OFFS = [0.1, 1, 5, 20]          # representative dark durations for figure A
# ordinal blue ramp, light->dark = short->long ON (monotone L, single hue, validated)
COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
# the sensitivity has a sign (rho can fall as the memory grows), so the heatmap needs a
# diverging scale: two poles and a neutral grey midpoint at zero
DIV = LinearSegmentedColormap.from_list(
    "div", ["#8f3a12", "#c2571f", "#eb6834", "#f6b393", "#eeedea",
            "#9ec5f4", "#3987e5", "#256abf", "#0d366b"])

d = {(float(x["tau_m_s"]), float(x["on_s"]), float(x["off_s"])):
     (float(x["win_mean"]), float(x["win_std"]))
     for x in csv.DictReader(open(f"{FIGS}/pulse_ratio.csv"))}

plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.0,
                     "xtick.direction": "in", "ytick.direction": "in"})

# ---------------------------------------------------------------- figure A
fig, axes = plt.subplots(1, len(PANEL_OFFS), figsize=(3.7*len(PANEL_OFFS), 5.2),
                         sharey=True)
for ax, off in zip(axes, PANEL_OFFS):
    for on, col in zip(ONS, COLORS):
        v = [d[(t, on, off)][0] for t in TAUS]
        e = [d[(t, on, off)][1] for t in TAUS]
        ax.errorbar(TAUS, v, yerr=e, color=col, lw=2.0, marker="o", ms=7.5,
                    mec="white", mew=1.2, capsize=0, elinewidth=2.0, zorder=3)
    ax.set(xscale="log", yscale="log", ylim=(1.0, 20), xlabel=r"$\tau_m$ (s)")
    ax.set_xticks(TAUS); ax.set_xticklabels([f"{t:g}" for t in TAUS])
    ax.minorticks_off()
    ax.set_title(f"dark {off:g} s")
    ax.grid(True, which="major", color="#e6e6e2", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel(r"$\rho_{\rm in}\,/\,\rho_{\rm out}$")
handles = [plt.Line2D([], [], color=c, lw=2.0, marker="o", ms=7.5, mec="white",
                      mew=1.2, label=f"{on} s ON") for on, c in zip(ONS, COLORS)]
fig.legend(handles=handles, frameon=False, ncol=len(ONS), fontsize=11,
           loc="upper center", bbox_to_anchor=(0.5, 0.95))
fig.suptitle("A flat line means the density cannot tell the memories apart",
             y=0.995, fontsize=13)
fig.tight_layout(rect=(0, 0.03, 1, 0.89))
fig.text(0.01, 0.01, "bars: seed-to-seed spread (5 seeds)", ha="left", va="bottom",
         color="#52514e", fontsize=10)
fig.savefig(f"{FIGS}/en/sens_rho_vs_tau.png", dpi=200)

# ---------------------------------------------------------------- figure B
lt_all = np.log(TAUS)
lt_top = np.log([16.0, 20.0])
S_all = np.zeros((len(ONS), len(OFFS)))
S_top = np.zeros_like(S_all)
for i, on in enumerate(ONS):
    for j, off in enumerate(OFFS):
        y = np.log([d[(t, on, off)][0] for t in TAUS])
        S_all[i, j] = np.polyfit(lt_all, y, 1)[0]
        S_top[i, j] = (y[3] - y[2])/(lt_top[1] - lt_top[0])

# per-cell uncertainty on the slope, propagated from the seed scatter of each mean
S_all_e = np.zeros_like(S_all); S_top_e = np.zeros_like(S_all)
for i, on in enumerate(ONS):
    for j, off in enumerate(OFFS):
        rel = np.array([d[(t, on, off)][1]/(d[(t, on, off)][0]*np.sqrt(5)) for t in TAUS])
        # slope of a least-squares fit in log-log: sum(w_k * log rho_k), w from the design
        w = (lt_all - lt_all.mean())/((lt_all - lt_all.mean())**2).sum()
        S_all_e[i, j] = np.sqrt(((w*rel)**2).sum())
        S_top_e[i, j] = np.sqrt(rel[3]**2 + rel[2]**2)/(lt_top[1] - lt_top[0])

vmax = np.abs(np.concatenate([S_all, S_top])).max()
fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.0))
for ax, S, E, lab in ((axes[0], S_all, S_all_e, r"full range,  $\tau_m$ 4 $\to$ 20 s"),
                      (axes[1], S_top, S_top_e, r"closest pair,  $\tau_m$ 16 $\to$ 20 s")):
    im = ax.imshow(S, cmap=DIV, vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
    ax.set_xticks(range(len(OFFS))); ax.set_xticklabels([f"{o:g}" for o in OFFS])
    ax.set_yticks(range(len(ONS)));  ax.set_yticklabels([f"{o:g}" for o in ONS])
    ax.set(xlabel="dark duration (s)", ylabel="light duration (s)")
    ax.set_title(f"sensitivity to memory — {lab}", fontsize=12)
    nsig = 0
    for i in range(len(ONS)):
        for j in range(len(OFFS)):
            sig = abs(S[i, j]) >= 2*E[i, j]
            nsig += sig
            if not sig:                      # non distinguibile da zero: tratteggiata
                ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False, lw=0,
                                           hatch="////", edgecolor="#ffffff", zorder=2))
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", zorder=3,
                    fontsize=9, fontweight="bold" if sig else "normal",
                    color="white" if abs(S[i, j]) > 0.6*vmax else "#2b2b2b")
    ax.tick_params(length=0)
    print(f"  {lab}: {nsig}/{S.size} celle significative a 2 sigma")
fig.subplots_adjust(left=0.055, right=0.90, bottom=0.17, top=0.84, wspace=0.16)
cax = fig.add_axes([0.925, 0.17, 0.012, 0.67])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$d\log\rho\ /\ d\log\tau_m$")
cb.outline.set_visible(False)
fig.suptitle("Where the density carries information about the memory", y=0.99, fontsize=13)
fig.text(0.005, 0.015, "White hatching: slope not distinguishable from zero at 2σ "
         "(σ propagated from the spread over 5 seeds). Zero = density blind to memory.",
         ha="left", va="bottom", color="#52514e", fontsize=10)
fig.savefig(f"{FIGS}/en/sens_heatmap.png", dpi=200)

print("scritte sens_rho_vs_tau.png e sens_heatmap.png in", FIGS)
print("pendenza massima full-range %.2f, massima 16->20 %.2f" % (S_all.max(), S_top.max()))
print("riga ON=1 s, pendenze full-range:", " ".join(f"{v:.2f}" for v in S_all[0]))
