"""P(theta) decomposed by the sign state of the gradient response.

The total P(theta) looks isotropic, but that is a cancellation: particles whose response
has not reversed swim up the gradient, the reversed ones swim down it, with nearly equal
and opposite bias. This figure shows the two sub-populations separately and the total they
add up to, then summarises the cancellation across every condition available.

theta is the angle between the swimming direction and the local illumination gradient,
as in the experimental pipeline; here it is taken from the logged orientation phi (see
theta_dist.py for why the experimental 0.5 s velocity window cannot be reproduced yet).
"""
import subprocess
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig_lang  # noqa: F401  Italian labels when FIG_LANG=it
from matplotlib.gridspec import GridSpec

OUT       = Path("/media/user/LaCie/pulsed_exps")
FIGS      = OUT/"figs"
ILL_MAP   = Path("tbr_fields/ill_map_ring_180muW.dat")
NPART     = 371
WINDOW    = 512
BINS      = 18
C_NORM    = "#2a78d6"     # categorical slot 1
C_REV     = "#eb6834"     # categorical slot 2
C_TOT     = "#52514e"     # neutral ink: the total is a reference, not a third category


def load_gradient(mapfile):
    with open(mapfile) as f:
        Nx, Ny, _, _, pix = f.readline().split()
        Nx, Ny, pix = int(Nx), int(Ny), float(pix)
        m = np.loadtxt(f)
    gx, gy = np.gradient(m, pix)
    return gx, gy, Nx, Ny, pix


def sample(run, gx, gy, Nx, Ny, pix):
    """cos(theta) and the reversed-state flag for every logged particle-frame."""
    f = next((OUT/run/"logs").glob("Simlog*"))
    tail = subprocess.run(["tail", "-n", str(NPART*WINDOW), str(f)],
                          capture_output=True, text=True).stdout.splitlines()
    d = np.loadtxt(tail[len(tail) % NPART:])
    x, y, phi, rev = d[:, 2]*10, d[:, 3]*10, d[:, 4], d[:, 6] < 0
    ix = np.clip(((x + Nx*pix/2)/pix).astype(int), 0, Nx-1)
    iy = np.clip(((y + Ny*pix/2)/pix).astype(int), 0, Ny-1)
    Gx, Gy = gx[ix, iy], gy[ix, iy]
    G = np.sqrt(Gx*Gx + Gy*Gy)
    ok = G > 0
    c = (np.cos(phi[ok])*Gx[ok] + np.sin(phi[ok])*Gy[ok])/G[ok]
    return np.clip(c, -1, 1), rev[ok]


def run_name(th, tau, on, off):
    return (f"tau{tau}_on{on}_off{off}_seed0" if th == "3.0708"
            else f"thr{float(th):g}_tau{tau}_on{on}_off{off}_seed0")


if __name__ == "__main__":
    gx, gy, Nx, Ny, pix = load_gradient(ILL_MAP)
    edges = np.linspace(0, np.pi, BINS+1)
    centres = 0.5*(edges[1:] + edges[:-1])

    # panels chosen to span the reversed fraction, from the mechanism off to mostly on
    panels = [("3.2",    16, 15, 1),
              ("3.0708", 8,  10, 1),
              ("3.0708", 8,  15, 1),
              ("2.6",    16, 15, 1)]

    # summary over the whole threshold scan
    summary = []
    for th in ["2.6", "2.8", "3", "3.0708", "3.2"]:
        for tau in (8, 16):
            for on in (10, 15):
                for off in (1, 5, 10):
                    try:
                        c, rev = sample(run_name(th, tau, on, off), gx, gy, Nx, Ny, pix)
                    except (StopIteration, OSError):
                        continue
                    f = rev.mean()
                    summary.append((f,
                                    c[~rev].mean() if (~rev).any() else np.nan,
                                    c[rev].mean() if rev.any() else np.nan,
                                    c.mean()))
    summary = np.array(summary)

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.0})
    fig = plt.figure(figsize=(15, 8.4))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1.05], hspace=0.42, wspace=0.12)

    for k, (th, tau, on, off) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        c, rev = sample(run_name(th, tau, on, off), gx, gy, Nx, Ny, pix)
        theta = np.arccos(c)
        frac = rev.mean()
        for mask, col, lab in ((~rev, C_NORM, "not reversed"), (rev, C_REV, "reversed")):
            if mask.any():
                h, _ = np.histogram(theta[mask], bins=edges, density=True)
                ax.plot(centres, h, color=col, lw=2.0, label=lab)
        h, _ = np.histogram(theta, bins=edges, density=True)
        ax.plot(centres, h, color=C_TOT, lw=2.0, ls=(0, (5, 2)), label="total")
        ax.axhline(1/np.pi, color="#9a9a95", lw=1.0, ls=(0, (2, 3)))
        ax.set(xlim=(0, np.pi), ylim=(0.13, 0.53))
        ax.set_xticks([0, np.pi/2, np.pi]); ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
        ax.set_xlabel(r"$\theta$")
        ax.set_title(f"threshold {th}, τ={tau} s, ON {on} s\nreversed: {100*frac:.0f}%",
                     fontsize=10.5)
        ax.grid(True, color="#e6e6e2", lw=0.8); ax.set_axisbelow(True)
        if k == 0:
            ax.set_ylabel(r"$P(\theta)$")
            # legenda da handle fissi: il primo pannello non ha particelle invertite
            ax.legend(handles=[plt.Line2D([], [], color=C_NORM, lw=2.0, label="not reversed"),
                               plt.Line2D([], [], color=C_REV, lw=2.0, label="reversed"),
                               plt.Line2D([], [], color=C_TOT, lw=2.0, ls=(0, (5, 2)),
                                          label="total")],
                      frameon=False, fontsize=9.5, loc="upper right")
        else:
            ax.tick_params(labelleft=False)

    ax = fig.add_subplot(gs[1, :])
    o = np.argsort(summary[:, 0])
    ax.axhline(0, color="#9a9a95", lw=1.0, ls=(0, (4, 3)))
    ax.plot(100*summary[o, 0], summary[o, 1], "o", ms=7, mec="white", mew=1.0,
            color=C_NORM, label="not reversed")
    ax.plot(100*summary[o, 0], summary[o, 2], "o", ms=7, mec="white", mew=1.0,
            color=C_REV, label="reversed")
    ax.plot(100*summary[o, 0], summary[o, 3], "o", ms=7, mec="white", mew=1.0,
            color=C_TOT, label="whole population")
    ax.set(xlabel="fraction of samples in the reversed state (%)",
           ylabel=r"$\langle\cos\theta\rangle$")
    ax.set_title("the two states are polarised in opposite directions; "
                 "the total stays at zero", fontsize=11)
    ax.grid(True, color="#e6e6e2", lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=10, loc="upper left")
    ax.text(0.995, 0.04, "positive = swimming towards the light", transform=ax.transAxes,
            ha="right", va="bottom", color="#52514e", fontsize=10)

    fig.suptitle(r"$P(\theta)$ split by the sign state of the gradient response",
                 y=0.975, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGS.mkdir(exist_ok=True)
    (FIGS/"en").mkdir(exist_ok=True)
    fig.savefig(FIGS/"en"/"theta_decomposed.png", dpi=180)
    print(f"{len(summary)} condizioni nel pannello di sintesi")
    print("range <cos> non invertite: %+.3f .. %+.3f" % (np.nanmin(summary[:, 1]), np.nanmax(summary[:, 1])))
    print("range <cos> invertite:     %+.3f .. %+.3f" % (np.nanmin(summary[:, 2]), np.nanmax(summary[:, 2])))
    print("range <cos> totale:        %+.3f .. %+.3f" % (np.nanmin(summary[:, 3]), np.nanmax(summary[:, 3])))
    print(f"scritta {FIGS}/theta_decomposed.png")
