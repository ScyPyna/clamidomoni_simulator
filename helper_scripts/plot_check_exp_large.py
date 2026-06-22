import numpy as np
import matplotlib.pyplot as plt

data = np.load("dataset_exp_largering/dataset.npy", allow_pickle=True)
r    = np.load("dataset_exp_largering/r_centers.npy")

# raggruppa per tau_m, media sui seed
from collections import defaultdict
by_tau = defaultdict(list)
for s in data:
    by_tau[s["tau_m"]].append(s["rho"])

fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
axes = axes.flatten()

for ax, (tau, rhos) in zip(axes, sorted(by_tau.items())):
    rho_mean = np.mean(rhos, axis=0)  # media sui seed
    # plot snapshot a t=0, t_metà, t_finale
    n = rho_mean.shape[1]
    for idx, label in [(0,"t=0"), (n//4,"t=1/4"), (n//2,"t=1/2"), (-1,"t=fine")]:
        ax.plot(r, rho_mean[:, idx], label=label)
    ax.set_title(f"τₘ = {tau:.1f} s")
    ax.set_xlabel("r (µm)")
    ax.set_ylabel("ρ/ρ₀")
    ax.legend(fontsize=7)
    ax.axvline(281, color='gray', linestyle='--', alpha=0.5, label='d_max')
    ax.axvline(134, color='red',  linestyle='--', alpha=0.5, label='d_thr')

plt.tight_layout()
plt.savefig("dataset_exp_largering/density_check.png", dpi=120)
print("Salvato dataset_exp_largering/density_check.png")
