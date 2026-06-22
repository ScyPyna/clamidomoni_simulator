import numpy as np
import os
from pathlib import Path

# ── configurazione ──────────────────────────────────────────
LOGS_DIR    = Path("logs_exp_smallring")
OUT_DIR     = Path("dataset_exp_smallring")
N_BINS      = 50
R_MAX       = 900.0
HEADER_ROWS = 2
PART_DIAM   = 10.0
# ────────────────────────────────────────────────────────────

def parse_simlog(filepath):
    name  = filepath.stem
    tmem  = int(name.split("tmem")[1].split("_")[0])
    tau_m = tmem * 0.0001

    data   = np.loadtxt(filepath, skiprows=HEADER_ROWS)
    t_vals = np.unique(data[:, 0])
    n_time = len(t_vals)
    n_part = int(data[:, 1].max()) + 1

    t     = t_vals
    x     = data[:, 2].reshape(n_time, n_part).T * PART_DIAM
    y     = data[:, 3].reshape(n_time, n_part).T * PART_DIAM
    phi   = data[:, 4].reshape(n_time, n_part).T
    trate = data[:, 5].reshape(n_time, n_part).T
    state = data[:, 6].reshape(n_time, n_part).T

    return dict(t=t, x=x, y=y, phi=phi,
                trate=trate, state=state, tau_m=tau_m)


def compute_density_profile(x, y, t, n_bins=N_BINS, r_max=R_MAX):
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    n_time = len(t)
    rho    = np.zeros((n_bins, n_time))

    for i in range(n_time):
        r_i = np.sqrt(x[:, i]**2 + y[:, i]**2)
        counts, _ = np.histogram(r_i, bins=bin_edges)
        rho[:, i] = counts / bin_areas

    rho0 = rho[:, 0].mean()
    if rho0 > 0:
        rho /= rho0

    return r_centers, rho


def compute_polarization(x, y, phi, t, n_bins=N_BINS, r_max=R_MAX):
    """
    Calcola T_r(r,t) = componente radiale media della polarizzazione
    T_r = <cos(φᵢ - θᵢ)> dove θᵢ è l'angolo radiale della particella
    """
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    n_time    = len(t)
    T_r       = np.zeros((n_bins, n_time))

    for i in range(n_time):
        r_i     = np.sqrt(x[:, i]**2 + y[:, i]**2)
        theta_i = np.arctan2(y[:, i], x[:, i])
        cos_diff = np.cos(phi[:, i] - theta_i)

        for b in range(n_bins):
            mask = (r_i >= bin_edges[b]) & (r_i < bin_edges[b+1])
            if mask.sum() > 0:
                T_r[b, i] = cos_diff[mask].mean()

    return T_r


def process_all():
    OUT_DIR.mkdir(exist_ok=True)
    files = sorted(LOGS_DIR.glob("traj_tmem*_seed*.dat"))
    print(f"Trovati {len(files)} file")

    all_samples = []

    for fpath in files:
        print(f"  elaboro {fpath.name} ...", end=" ", flush=True)
        sim      = parse_simlog(fpath)
        r_c, rho = compute_density_profile(sim["x"], sim["y"], sim["t"])
        T_r      = compute_polarization(sim["x"], sim["y"],
                                        sim["phi"], sim["t"])

        sample = {
            "tau_m"    : sim["tau_m"],
            "t"        : sim["t"],
            "r_centers": r_c,
            "rho"      : rho,
            "T_r"      : T_r,
            "x"        : sim["x"],
            "y"        : sim["y"],
            "phi"      : sim["phi"],
            "state"    : sim["state"],
        }
        all_samples.append(sample)
        print(f"τₘ={sim['tau_m']:.1f}s  rho={rho.shape}  T_r={T_r.shape}")

    out_path = OUT_DIR / "dataset.npy"
    np.save(out_path, all_samples, allow_pickle=True)
    print(f"\nDataset salvato in {out_path}")
    print(f"Totale campioni: {len(all_samples)}")

    np.save(OUT_DIR / "r_centers.npy", r_c)


if __name__ == "__main__":
    process_all()
