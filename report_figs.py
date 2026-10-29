"""Extra figures for the English summary report (2026-09-27).

  lowpass.png        rho_in/rho_out vs cycle period at 50 % duty cycle, one line per tau_m
  exp_vs_sim.png     lab data (read off the lab figure, approximate) vs simulations, in the lab
                     convention f = N_in / (N_in + N_out), where uniform density is f = 0.105
  thr_scan.png       ON=10 / ON=15 accumulation ratio vs grad_sign_thresh, with the lab value

Reads figs/pulse_ratio.csv (written by pulse_ratio.py) and, for the threshold scan, the tails
of the logs. Per-run threshold-scan numbers are cached in figs/thr_scan_cache.csv; runs whose
log is incomplete are skipped and not cached, so re-running after batch 5 picks them up.
"""
import csv, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_lang  # noqa: F401  Italian labels when FIG_LANG=it

OUT   = Path("/media/user/LaCie/pulsed_exps")
FIGS  = OUT/"figs"
EN    = FIGS/"en"
CACHE = FIGS/"thr_scan_cache.csv"
NPART = 371
WINDOW = 512
L_BOX, R_RING = 1207.3, 221.0                 # microns, same geometry as pulse_ratio.py
A_IN = np.pi*R_RING**2
A_OUT = L_BOX**2 - A_IN
AREA = A_OUT/A_IN                              # 8.52: density ratio -> count fraction
TAUS = [4, 8, 16, 20]
TAU_COL = {4: "#9fd4b4", 8: "#55a97f", 16: "#23734f", 20: "#0b412b"}
INK, MUTED, GRID = "#2b2b2b", "#6b6b66", "#e6e6e2"

# lab data, light on 1/5/10/15 s, dark 0.01/0.1/1/5/10 s, count fraction N_in/(N_in+N_out),
# read off the lab figure by eye (the 5 s series is considered unreliable by the lab)
EXP_OFFS = [0.01, 0.1, 1, 5, 10]
EXP = {1: [0.66, 0.37, 0.17, None, None],
       10: [0.61, 0.53, 0.47, 0.25, 0.15],
       15: [0.51, 0.34, 0.32, 0.12, 0.11]}
def _dens(f):                                   # count fraction -> rho_in/rho_out (area cancels in ratios)
    return f/(1 - f)
EXP_RATIO_10_15 = {off: _dens(EXP[10][EXP_OFFS.index(off)])/_dens(EXP[15][EXP_OFFS.index(off)])
                   for off in (1, 5, 10)}      # 1.88, 2.44, 1.43

plt.rcParams.update({"font.size": 11.5, "axes.linewidth": 1.0, "axes.edgecolor": "#55554f",
                     "xtick.direction": "in", "ytick.direction": "in"})


def style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.minorticks_off()


agg = {}
for r in csv.DictReader(open(FIGS/"pulse_ratio.csv")):
    agg[(int(float(r["tau_m_s"])), float(r["on_s"]), float(r["off_s"]))] = r

# ------------------------------------------------------------------ low-pass filter
fig, ax = plt.subplots(figsize=(7.2, 4.8))
pairs = [(1, 1), (5, 5), (10, 10), (20, 20)]
for tau in TAUS:
    P = [on + off for on, off in pairs]
    v = [float(agg[(tau, on, off)]["win_mean"]) for on, off in pairs]
    e = [float(agg[(tau, on, off)]["win_std"]) for on, off in pairs]
    ax.errorbar(P, v, yerr=e, color=TAU_COL[tau], lw=2.2, marker="o", ms=8, mec="white",
                mew=1.2, elinewidth=2, capsize=0, zorder=3, label=rf"$\tau_m$ = {tau} s")
    change = 100*(max(v) - min(v))/max(v)
    ax.annotate(f"{change:.0f} %", (P[-1], v[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=10, color=TAU_COL[tau])
ax.set(xscale="log", xlabel="cycle period (s), light on half of the time",
       ylabel=r"$\rho_{\rm in}\,/\,\rho_{\rm out}$  (mean over last 419 s)", xlim=(1.6, 60))
ax.set_xticks([2, 10, 20, 40]); ax.set_xticklabels(["2", "10", "20", "40"])
style(ax)
ax.legend(frameon=False, fontsize=10, loc="lower left")
ax.set_title("Same light dose, different rhythm: long memories average the rhythm out",
             fontsize=11.5)
fig.text(0.01, 0.01, "labels at right: change from 2 s to 40 s period    |    bars: seed-to-seed spread",
         color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(EN/"lowpass.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ experiment vs simulation
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), sharey=True)
for ax, on in zip(axes, (1, 10, 15)):
    ax.axhline(1/(1 + AREA), color="#9a9a95", lw=1.0, ls=(0, (4, 3)), zorder=1)
    for tau in TAUS:
        v = [float(agg[(tau, on, o)]["tend_mean"]) for o in EXP_OFFS]
        f = [x/(x + AREA) for x in v]
        ax.plot(EXP_OFFS, f, color=TAU_COL[tau], lw=2.0, marker="o", ms=6, mec="white",
                mew=1.0, zorder=3, label=rf"sim, $\tau_m$ = {tau} s")
    ex = [(o, y) for o, y in zip(EXP_OFFS, EXP[on]) if y is not None]
    ax.plot(*zip(*ex), ls="none", marker="D", ms=8.5, color=INK, mec="white", mew=1.2,
            zorder=4, label="lab (read off figure)")
    ax.set(xscale="log", xlabel="dark duration (s)", ylim=(0, 0.95))
    ax.set_xticks(EXP_OFFS); ax.set_xticklabels([f"{o:g}" for o in EXP_OFFS])
    ax.set_title(f"light on {on} s")
    style(ax)
axes[0].set_ylabel(r"$N_{\rm in}\,/\,(N_{\rm in}+N_{\rm out})$")
axes[0].text(0.012, 1/(1 + AREA) + 0.015, "uniform = 0.105", color=MUTED, fontsize=9.5)
axes[2].legend(frameon=False, fontsize=9.5, loc="upper right")
fig.suptitle("Lab and model cover the same range; the model misses the 10 s > 15 s ordering",
             fontsize=12.5, y=0.99)
fig.text(0.01, 0.01, "Simulations: value at the end of the run, mean of 5 seeds. "
         "Lab values read by eye from the lab figure, 5 acquisitions per point.",
         color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(EN/"exp_vs_sim.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ threshold scan
def run_ratio(run):
    """Window-averaged density ratio and reversed fraction of one run, or None if unfinished."""
    logs = list((OUT/run/"logs").glob("Simlog*"))
    if not logs:
        return None
    f = logs[0]
    n = int(subprocess.run(["wc", "-l", str(f)], capture_output=True, text=True).stdout.split()[0])
    if n != 2 + 2049*NPART:
        return None
    t = subprocess.run(["tail", "-n", str(NPART*WINDOW), str(f)],
                       capture_output=True, text=True).stdout.splitlines()
    d = np.loadtxt(t[len(t) % NPART:])
    nf = len(d)//NPART
    r = np.hypot(d[:, 2]*10, d[:, 3]*10).reshape(nf, NPART)
    ni = (r < R_RING).sum(axis=1)
    return float(((ni/A_IN)/np.maximum(NPART - ni, 1)*A_OUT).mean()), float((d[:, 6] < 0).mean())


THRS = ["2.2", "2.4", "2.6", "2.8", "3", "3.0708", "3.2"]
SCAN_TAUS, SCAN_ONS, SCAN_OFFS = [4, 8, 16], [10, 15], [1, 5, 10]
DEPLETED = 1.0      # rho_in/rho_out below uniform: the ring is emptied, the 10/15 ratio is not shown


def name(th, tau, on, off, s):
    return (f"tau{tau}_on{on}_off{off}_seed{s}" if th == "3.0708"
            else f"thr{th}_tau{tau}_on{on}_off{off}_seed{s}")


cache = {}
if CACHE.exists():
    for r in csv.DictReader(open(CACHE)):
        cache[r["run"]] = (float(r["ratio"]), float(r["reversed"]))
new = 0
for th in THRS:
    for tau in SCAN_TAUS:
        for on in SCAN_ONS:
            for off in SCAN_OFFS:
                for s in range(5):
                    n = name(th, tau, on, off, s)
                    if n in cache or not (OUT/n).exists():
                        continue
                    v = run_ratio(n)
                    if v is not None:
                        cache[n] = v; new += 1
with open(CACHE, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run", "ratio", "reversed"])
    for k, (a, b) in sorted(cache.items()):
        w.writerow([k, f"{a:.6f}", f"{b:.6f}"])
print(f"threshold scan: {len(cache)} runs cached ({new} new)")

rows = []
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), sharey=True)
for ax, off in zip(axes, SCAN_OFFS):
    ax.axhline(1.0, color="#9a9a95", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axhline(EXP_RATIO_10_15[off], color=INK, lw=1.6, zorder=2)
    ax.text(3.24, EXP_RATIO_10_15[off], "lab", va="bottom", ha="right", fontsize=9.5, color=INK)
    for tau in SCAN_TAUS:
        xs, ys, es = [], [], []
        for th in THRS:
            a = [cache.get(name(th, tau, 10, off, s)) for s in range(5)]
            b = [cache.get(name(th, tau, 15, off, s)) for s in range(5)]
            a = np.array([x[0] for x in a if x]); b = np.array([x[0] for x in b if x])
            if len(a) < 3 or len(b) < 3:
                continue
            if min(a.mean(), b.mean()) < DEPLETED:   # inside emptied: a ratio of near-zero values
                ax.plot(float(th), 0.35, marker="x", ms=8, mew=2, color=TAU_COL[tau], zorder=3, ls="none")
                continue
            q = a.mean()/b.mean()
            err = q*np.hypot(a.std(ddof=1)/a.mean()/np.sqrt(len(a)), b.std(ddof=1)/b.mean()/np.sqrt(len(b)))
            xs.append(float(th)); ys.append(q); es.append(err)
            rows.append((th, tau, off, q, err))
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=TAU_COL[tau], lw=2.0, marker="o", ms=7, mec="white",
                        mew=1.1, capsize=0, elinewidth=1.8, zorder=3, label=rf"$\tau_m$ = {tau} s")
    ax.axvline(3.0708, color="#c9c9c3", lw=6, zorder=0)
    ax.set(xlabel="grad_sign_thresh", xlim=(2.1, 3.3), title=f"dark {off} s")
    style(ax)
axes[0].set_ylabel(r"$\rho(\mathrm{ON}\ 10\,\mathrm{s})\ /\ \rho(\mathrm{ON}\ 15\,\mathrm{s})$")
axes[1].legend(frameon=False, fontsize=10, loc="upper right", bbox_to_anchor=(1, 0.93))
axes[1].text(3.0708, 0.03, "current\nvalue", transform=axes[1].get_xaxis_transform(),
             ha="center", fontsize=8.5, color=MUTED)
fig.suptitle("Low thresholds give the lab ordering (10 s > 15 s), but at 1 s dark they empty the ring",
             fontsize=12.5, y=0.99)
fig.text(0.01, 0.01, "Above 1: 10 s of light accumulates more than 15 s. "
         "Bars: standard error of the ratio of 5-seed means. Crosses at the bottom: inside of the ring emptied "
         "(rho_in/rho_out < 1), ratio not shown.",
         color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(EN/"thr_scan.png", dpi=200)
plt.close(fig)
with open(FIGS/"thr_scan.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["threshold", "tau_m_s", "off_s", "ratio_on10_on15", "err"])
    w.writerows([(a, b, c, f"{d:.4f}", f"{e:.4f}") for a, b, c, d, e in rows])
for r in rows:
    print("  thr %-6s tau %2d off %2d  ratio %.2f +- %.2f" % r)
