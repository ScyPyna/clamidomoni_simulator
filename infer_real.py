import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

# ── configurazione ────────────────────────────────────────────
BASE = "/media/user/LaCie/Videos/Novembre2024"

GROUPS = {
    "large ring ill24": {
        "files": [f"{BASE}/tAlgae{i}.txt" for i in range(414, 428)],
        "r_out": 371.5,
        "pix"  : 1/0.38,
    },
    "small ring ill25": {
        "files": [f"{BASE}/tAlgae{i}.txt" for i in range(428, 433)],
        "r_out": 247.1,
        "pix"  : 1/0.96,
    },
}

FPS        = 10.0
TAU_MAX    = 20.0
N_BINS     = 50
N_BINS_VIZ = 20
R_MAX      = 900.0
N_TIME     = 2049
MODEL_PATH = "dataset/best_model_all.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────

def load_trajectories(filepath, pix_um):
    data      = np.loadtxt(filepath, delimiter=',', skiprows=1)
    frames    = data[:, 0].astype(int)
    y_px      = data[:, 1]
    x_px      = data[:, 2]
    particles = data[:, 3].astype(int)
    x_um = x_px * pix_um
    y_um = y_px * pix_um
    x_um -= (x_um.max() + x_um.min()) / 2
    y_um -= (y_um.max() + y_um.min()) / 2
    return frames, x_um, y_um, particles


def compute_density_profile(frames, x_um, y_um, fps,
                             n_bins=N_BINS, n_bins_viz=N_BINS_VIZ,
                             r_max=R_MAX):
    frame_ids = np.unique(frames)
    n_time    = len(frame_ids)
    t_vals    = frame_ids / fps

    edges_viz = np.linspace(0, r_max, n_bins_viz + 1)
    areas_viz = np.pi * (edges_viz[1:]**2 - edges_viz[:-1]**2)
    r_viz     = 0.5 * (edges_viz[:-1] + edges_viz[1:])
    rho_viz   = np.zeros((n_bins_viz, n_time))

    for i, fid in enumerate(frame_ids):
        mask = frames == fid
        r_i  = np.sqrt(x_um[mask]**2 + y_um[mask]**2)
        counts, _ = np.histogram(r_i, bins=edges_viz)
        rho_viz[:, i] = counts / areas_viz

    rho0 = rho_viz[:, 0].mean()
    if rho0 > 0:
        rho_viz /= rho0

    r_net   = np.linspace(r_viz[0], r_viz[-1], n_bins)
    rho_net = np.zeros((n_bins, n_time))
    for t in range(n_time):
        f = interp1d(r_viz, rho_viz[:, t], kind='linear',
                     fill_value='extrapolate')
        rho_net[:, t] = np.clip(f(r_net), 0, None)

    return t_vals, rho_net, r_viz, rho_viz


def adapt_time(rho, n_time_target=N_TIME):
    n_bins, n_time = rho.shape
    if n_time == n_time_target:
        return rho
    t_old   = np.linspace(0, 1, n_time)
    t_new   = np.linspace(0, 1, n_time_target)
    rho_out = np.zeros((n_bins, n_time_target))
    for b in range(n_bins):
        f = interp1d(t_old, rho[b, :], kind='linear',
                     fill_value='extrapolate')
        rho_out[b, :] = f(t_new)
    return rho_out


# ── carica modello ────────────────────────────────────────────
from pinn_all import ChlamyPINN
model = ChlamyPINN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ── inferenza ─────────────────────────────────────────────────
print(f"\nDevice: {device}")
print("── Inferenza τₘ su dati sperimentali ──────────────────")

results = {}

for group_label, cfg in GROUPS.items():
    print(f"\n{'='*50}")
    print(f"{group_label}")
    tau_list = []
    rho_list = []

    for fpath in cfg["files"]:
        if not Path(fpath).exists():
            print(f"  MANCANTE: {fpath}")
            continue
        fname = Path(fpath).name
        try:
            frames, x_um, y_um, particles = load_trajectories(fpath, cfg["pix"])
            t_vals, rho_net, r_viz, rho_viz = compute_density_profile(
                frames, x_um, y_um, FPS)

            rho_input = adapt_time(rho_net)
            rho_t   = torch.tensor(rho_input, dtype=torch.float32).unsqueeze(0).to(device)
            r_out_t = torch.tensor([cfg["r_out"] / 420.0],
                                    dtype=torch.float32).to(device)
            with torch.no_grad():
                tau_pred = model(rho_t, r_out_t).item() * TAU_MAX

            tau_list.append(tau_pred)
            rho_list.append(rho_viz)
            print(f"  {fname}: τₘ = {tau_pred:.2f} s")

        except Exception as e:
            print(f"  {fname}: ERRORE — {e}")

    if tau_list:
        tau_arr = np.array(tau_list)
        print(f"\n  → τₘ medio:   {tau_arr.mean():.2f} s")
        print(f"  → τₘ std:     {tau_arr.std():.2f} s")
        print(f"  → τₘ mediana: {np.median(tau_arr):.2f} s")
        results[group_label] = {
            "tau_list": tau_list,
            "rho_mean": np.mean(rho_list, axis=0),
            "r_viz"   : r_viz,
            "cfg"     : cfg,
        }

# ── plot riassuntivo ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for idx, (group_label, res) in enumerate(results.items()):
    cfg      = res["cfg"]
    rho_mean = res["rho_mean"]
    r_viz    = res["r_viz"]
    tau_list = res["tau_list"]
    tau_arr  = np.array(tau_list)
    tau_mean = tau_arr.mean()
    tau_std  = tau_arr.std()
    n        = rho_mean.shape[1]

    # profilo medio
    ax = axes[idx, 0]
    for i, lab in [(0,'t=0'), (n//4,'t=1/4'), (n//2,'t=1/2'), (-1,'t=fine')]:
        ax.plot(r_viz, rho_mean[:, i], label=lab)
    ax.axvline(cfg["r_out"], color='red', ls='--', alpha=0.7,
               label=f'r_out={cfg["r_out"]}µm')
    ax.set_xlabel('r (µm)')
    ax.set_ylabel('ρ/ρ₀')
    ax.set_title(f'{group_label} — profilo medio\nτₘ = {tau_mean:.2f} ± {tau_std:.2f} s')
    ax.legend(fontsize=8)

    # distribuzione τₘ
    ax = axes[idx, 1]
    ax.hist(tau_list, bins=min(10, len(tau_list)),
            color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(tau_mean, color='red', ls='--', label=f'media={tau_mean:.2f}s')
    ax.axvline(tau_mean - tau_std, color='orange', ls=':', alpha=0.7)
    ax.axvline(tau_mean + tau_std, color='orange', ls=':', alpha=0.7,
               label=f'±σ={tau_std:.2f}s')
    ax.set_xlabel('τₘ predetto (s)')
    ax.set_ylabel('conteggio')
    ax.set_title(f'{group_label} — distribuzione τₘ')
    ax.legend()

plt.tight_layout()
out = "dataset/inference_real_all.png"
plt.savefig(out, dpi=150)
print(f"\nSalvato: {out}")
