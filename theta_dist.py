"""P(theta) from the simulated trajectories, following the experimental definition.

theta = arccos( (v . grad I) / (|v| |grad I|) ), in [0, pi]: the angle between the
swimming direction and the local illumination gradient. theta = 0 means swimming up the
gradient (towards the light), theta = pi away from it. An isotropic population gives a
flat P(theta) = 1/pi = 0.318, which is the reference line on every panel.

The experimental pipeline (clamutils_exp.grad_theta) builds v as a finite difference over
particle_show=5 frames of dt=0.1 s, i.e. a 0.5 s window. Our logs are sampled every
0.82 s, too coarse for that, so here v is taken from the logged orientation phi instead:
the instantaneous heading, which the experiment cannot measure directly. The two are not
identical — the experimental theta is coarse-grained over half a second and therefore
narrower — so compare shapes between simulated conditions, not absolute widths against
the experiment.

Split radii follow the experiment: inside r <= 300 um, outside 300 < r < 800 um.
"""
import subprocess
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT        = Path("/media/user/LaCie/pulsed_exps")
FIGS       = OUT/"figs"
ILL_MAP    = Path("tbr_fields/ill_map_ring_180muW.dat")
PART_DIAM  = 10.0
NPART      = 371
WINDOW     = 512          # logged frames used, at the end of the run
R_INNER    = 300.0        # um, as in the experimental analysis
R_MAX      = 800.0
BINS       = 18
COLOR_IN   = "#1baf7a"    # the experiment's green/orange pairing
COLOR_OUT  = "#eb6834"


def load_gradient(mapfile):
    """Illumination map plus its gradient, and the microns-per-pixel scale."""
    with open(mapfile) as f:
        Nx, Ny, _, _, pix = f.readline().split()
        Nx, Ny, pix = int(Nx), int(Ny), float(pix)
        m = np.loadtxt(f)
    gx, gy = np.gradient(m, pix)          # m is indexed [ix, iy]
    return m, gx, gy, Nx, Ny, pix


def theta_samples(run, gx, gy, Nx, Ny, pix):
    """theta and radial distance for every particle in the final frames of a run."""
    f = next((OUT/run/"logs").glob("Simlog*"))
    tail = subprocess.run(["tail", "-n", str(NPART*WINDOW), str(f)],
                          capture_output=True, text=True).stdout.splitlines()
    d = np.loadtxt(tail[len(tail) % NPART:])
    x = d[:, 2]*PART_DIAM
    y = d[:, 3]*PART_DIAM
    phi = d[:, 4]

    ix = np.clip(((x + Nx*pix/2)/pix).astype(int), 0, Nx-1)
    iy = np.clip(((y + Ny*pix/2)/pix).astype(int), 0, Ny-1)
    Gx, Gy = gx[ix, iy], gy[ix, iy]
    G = np.sqrt(Gx*Gx + Gy*Gy)

    ok = G > 0
    cosT = (np.cos(phi[ok])*Gx[ok] + np.sin(phi[ok])*Gy[ok])/G[ok]
    return np.arccos(np.clip(cosT, -1, 1)), np.sqrt(x[ok]**2 + y[ok]**2)


def panel(ax, th, r, title):
    ins = th[r <= R_INNER]
    out = th[(r > R_INNER) & (r < R_MAX)]
    for data, col, lab in ((ins, COLOR_IN, f"dentro, r<{R_INNER:.0f} µm  (n={ins.size//1000}k)"),
                           (out, COLOR_OUT, f"fuori  (n={out.size//1000}k)")):
        if data.size:
            ax.hist(data, bins=BINS, range=(0, np.pi), density=True,
                    histtype="step", lw=2.0, color=col, label=lab)
    ax.axhline(1/np.pi, color="#9a9a95", lw=1.0, ls=(0, (4, 3)))
    ax.set_xticks([0, np.pi/2, np.pi]); ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$\theta$")
    ax.grid(True, color="#e6e6e2", lw=0.8); ax.set_axisbelow(True)
    # bias: mean cos(theta) > 0 means a net drift towards the light
    return (np.cos(ins).mean() if ins.size else np.nan,
            np.cos(out).mean() if out.size else np.nan)


if __name__ == "__main__":
    _, gx, gy, Nx, Ny, pix = load_gradient(ILL_MAP)
    conds = [("tau8_on10_off1_seed0",  "τ=8 s, ON 10 s, buio 1 s"),
             ("tau8_on15_off1_seed0",  "τ=8 s, ON 15 s, buio 1 s"),
             ("tau16_on10_off1_seed0", "τ=16 s, ON 10 s, buio 1 s"),
             ("tau16_on15_off1_seed0", "τ=16 s, ON 15 s, buio 1 s"),
             ("tau8_on10_off10_seed0", "τ=8 s, ON 10 s, buio 10 s"),
             ("tau8_on15_off10_seed0", "τ=8 s, ON 15 s, buio 10 s")]

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.0})
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharey=True)
    print("bias medio <cos θ>  (>0 = deriva verso la luce)")
    for ax, (run, title) in zip(axes.ravel(), conds):
        th, r = theta_samples(run, gx, gy, Nx, Ny, pix)
        bi, bo = panel(ax, th, r, title)
        print(f"  {title:28s} dentro {bi:+.3f}   fuori {bo:+.3f}")
    axes[0, 0].set_ylabel(r"$P(\theta)$"); axes[1, 0].set_ylabel(r"$P(\theta)$")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper center")
    fig.suptitle(r"$P(\theta)$ fra direzione di nuoto e gradiente di luce — "
                 r"tratteggio: isotropo ($1/\pi$)", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS/"theta_distribution.png", dpi=180)
    print(f"\nscritta {FIGS}/theta_distribution.png")
