"""Full grids of P(theta) decomposed by sign state, for visual scanning.

One figure per memory time for the main grid (rows = light duration, columns = dark
duration), plus one figure per memory time for the threshold scan (rows = threshold).
Every panel shares the same axes so shapes can be compared across the grid at a glance.

Blue = particles whose gradient response has not reversed, orange = reversed, dashed grey
= the whole population, dotted line = isotropic (1/pi). The number in each panel title is
the fraction of particle-frames in the reversed state.
"""
import subprocess
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_lang  # noqa: F401  Italian labels when FIG_LANG=it

OUT      = Path("/media/user/LaCie/pulsed_exps")
FIGS     = OUT/"figs"
ILL_MAP  = Path("tbr_fields/ill_map_ring_180muW.dat")
NPART    = 371
WINDOW   = 256            # logged frames per run: ~95k samples per panel, plenty for a shape
BINS     = 14
YLIM     = (0.13, 0.53)
C_NORM, C_REV, C_TOT = "#2a78d6", "#eb6834", "#52514e"

TAUS = [4, 8, 16, 20]
ONS  = [1, 5, 10, 15, 20]
OFFS = [0.01, 0.1, 0.5, 1, 2, 5, 7, 10, 15, 20]
THRS = ["2.2", "2.4", "2.6", "2.8", "3", "3.0708", "3.2"]
SCAN_ONS, SCAN_OFFS = [10, 15, 20], [1, 5, 10]


def load_gradient(mapfile):
    with open(mapfile) as f:
        Nx, Ny, _, _, pix = f.readline().split()
        Nx, Ny, pix = int(Nx), int(Ny), float(pix)
        m = np.loadtxt(f)
    gx, gy = np.gradient(m, pix)
    return gx, gy, Nx, Ny, pix


def sample(run, G):
    gx, gy, Nx, Ny, pix = G
    try:
        f = next((OUT/run/"logs").glob("Simlog*"))
    except StopIteration:
        return None, None
    tail = subprocess.run(["tail", "-n", str(NPART*WINDOW), str(f)],
                          capture_output=True, text=True).stdout.splitlines()
    if len(tail) < NPART:
        return None, None
    d = np.loadtxt(tail[len(tail) % NPART:])
    x, y, phi, rev = d[:, 2]*10, d[:, 3]*10, d[:, 4], d[:, 6] < 0
    ix = np.clip(((x + Nx*pix/2)/pix).astype(int), 0, Nx-1)
    iy = np.clip(((y + Ny*pix/2)/pix).astype(int), 0, Ny-1)
    Gx, Gy = gx[ix, iy], gy[ix, iy]
    Gn = np.sqrt(Gx*Gx + Gy*Gy)
    ok = Gn > 0
    c = (np.cos(phi[ok])*Gx[ok] + np.sin(phi[ok])*Gy[ok])/Gn[ok]
    return np.clip(c, -1, 1), rev[ok]


def draw_panel(ax, c, rev, edges, centres, label):
    if c is None:
        ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes,
                color="#9a9a95")
        frac = np.nan
    else:
        theta = np.arccos(c)
        frac = rev.mean()
        for mask, col in ((~rev, C_NORM), (rev, C_REV)):
            if mask.sum() > 200:
                h, _ = np.histogram(theta[mask], bins=edges, density=True)
                ax.plot(centres, h, color=col, lw=1.4)
        h, _ = np.histogram(theta, bins=edges, density=True)
        ax.plot(centres, h, color=C_TOT, lw=1.4, ls=(0, (4, 2)))
    ax.axhline(1/np.pi, color="#9a9a95", lw=0.8, ls=(0, (2, 3)))
    ax.set(xlim=(0, np.pi), ylim=YLIM, xticks=[0, np.pi/2, np.pi], yticks=[0.2, 0.3, 0.4])
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.grid(True, color="#eeeeea", lw=0.6); ax.set_axisbelow(True)
    if not np.isnan(frac):   # dentro al pannello: il titolo serve alle etichette di colonna
        ax.text(0.97, 0.95, f"{100*frac:.0f}%", transform=ax.transAxes, ha="right",
                va="top", fontsize=7.5, color="#52514e")


def grid_figure(rows, cols, runner, row_lab, col_lab, title, fname, G):
    edges = np.linspace(0, np.pi, BINS+1)
    centres = 0.5*(edges[1:] + edges[:-1])
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(1.45*len(cols)+1.2, 1.35*len(rows)+1.3),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows):
        for j, cl in enumerate(cols):
            c, rev = sample(runner(r, cl), G)
            draw_panel(axes[i, j], c, rev, edges, centres, "")
        axes[i, 0].set_ylabel(row_lab(r), fontsize=9)
        axes[i, 0].tick_params(labelleft=True, length=2)
        axes[i, 0].set_yticklabels(["0.2", "0.3", "0.4"], fontsize=7)
    for j, cl in enumerate(cols):
        axes[0, j].set_title(col_lab(cl), fontsize=9, pad=6)
        axes[-1, j].tick_params(labelbottom=True, length=2)
        axes[-1, j].set_xticklabels(["0", r"$\pi/2$", r"$\pi$"], fontsize=7)
    handles = [plt.Line2D([], [], color=C_NORM, lw=1.8, label="not reversed"),
               plt.Line2D([], [], color=C_REV, lw=1.8, label="reversed"),
               plt.Line2D([], [], color=C_TOT, lw=1.8, ls=(0, (4, 2)), label="total")]
    fig.legend(handles=handles, frameon=False, ncol=3, fontsize=9,
               loc="upper center", bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, y=0.995, fontsize=12)
    fig.supxlabel(r"$\theta$", fontsize=10)
    fig.supylabel(r"$P(\theta)$", fontsize=10)
    fig.tight_layout(rect=(0.012, 0.012, 1, 0.925))
    fig.savefig(FIGS/"en"/fname, dpi=170)
    plt.close(fig)
    print("  scritta", fname)


if __name__ == "__main__":
    G = load_gradient(ILL_MAP)
    (FIGS/"en").mkdir(parents=True, exist_ok=True)

    print("griglia principale (soglia 3.0708):")
    for tau in TAUS:
        grid_figure(
            ONS, OFFS,
            lambda on, off, t=tau: f"tau{t}_on{on}_off{off:g}_seed0",
            lambda on: f"ON {on} s", lambda off: f"dark {off:g} s",
            rf"$P(\theta)$ by sign state — $\tau_m$ = {tau} s, threshold 3.0708",
            f"theta_grid_tau{tau}.png", G)

    print("scan della soglia:")
    for tau in (4, 8, 16):
        cols = [(on, off) for on in SCAN_ONS for off in SCAN_OFFS]
        grid_figure(
            THRS, cols,
            lambda th, c, t=tau: (f"tau{t}_on{c[0]}_off{c[1]}_seed0" if th == "3.0708"
                                  else f"thr{float(th):g}_tau{t}_on{c[0]}_off{c[1]}_seed0"),
            lambda th: f"thr {th}", lambda c: f"ON {c[0]}\ndark {c[1]}",
            rf"$P(\theta)$ by sign state — threshold scan, $\tau_m$ = {tau} s",
            f"theta_grid_thr_tau{tau}.png", G)
