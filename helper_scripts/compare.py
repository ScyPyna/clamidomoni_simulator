import numpy as np
import matplotlib.pyplot as plt
import json

def parse_results(filename):
    """Legge i risultati dal log di training"""
    results = {
        "train_loss": [],
        "val_loss"  : [],
        "tau_true"  : [],
        "tau_pred"  : [],
        "rmse"      : None
    }
    
    with open(filename) as f:
        for line in f:
            # loss per epoca
            if "loss_tr=" in line:
                tr  = float(line.split("loss_tr=")[1].split("|")[0].strip())
                val = float(line.split("loss_val=")[1].split("|")[0].strip())
                results["train_loss"].append(tr)
                results["val_loss"].append(val)
            # risultati test
            if "τₘ vero:" in line:
                true = float(line.split("τₘ vero:")[1].split("s")[0].strip())
                pred = float(line.split("τₘ predetto:")[1].split("s")[0].strip())
                results["tau_true"].append(true)
                results["tau_pred"].append(pred)
            # RMSE
            if "RMSE:" in line:
                results["rmse"] = float(line.split("RMSE:")[1].split("s")[0].strip())
    
    return results

# ── carica risultati ─────────────────────────────────────────
r1 = parse_results("results_v1.txt")
r2 = parse_results("results_v2.txt")

print(f"V1 — RMSE: {r1['rmse']:.3f} s  |  campioni test: {len(r1['tau_true'])}")
print(f"V2 — RMSE: {r2['rmse']:.3f} s  |  campioni test: {len(r2['tau_true'])}")

# ── figura con 3 pannelli ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── pannello 1: training history ─────────────────────────────
ax = axes[0]
epochs1 = range(len(r1["train_loss"]))
epochs2 = range(len(r2["train_loss"]))
ax.plot(epochs1, r1["train_loss"], color="steelblue",  lw=1.5, label="V1 train")
ax.plot(epochs1, r1["val_loss"],   color="steelblue",  lw=1.5, ls="--", label="V1 val")
ax.plot(epochs2, r2["train_loss"], color="darkorange", lw=1.5, label="V2 train")
ax.plot(epochs2, r2["val_loss"],   color="darkorange", lw=1.5, ls="--", label="V2 val")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training history")
ax.legend()
ax.set_yscale("log")

# ── pannello 2: predetto vs vero ─────────────────────────────
ax = axes[1]
tau_range = [0, 21]
ax.plot(tau_range, tau_range, "k--", lw=1, alpha=0.5, label="perfetto")
ax.scatter(r1["tau_true"], r1["tau_pred"],
           color="steelblue", s=80, zorder=5, label=f"V1 (RMSE={r1['rmse']:.2f}s)")
ax.scatter(r2["tau_true"], r2["tau_pred"],
           color="darkorange", s=80, marker="^", zorder=5, label=f"V2 (RMSE={r2['rmse']:.2f}s)")
ax.set_xlabel("τₘ vero (s)")
ax.set_ylabel("τₘ predetto (s)")
ax.set_title("Predetto vs Vero — Test set")
ax.legend()
ax.set_xlim(tau_range)
ax.set_ylim(tau_range)

# ── pannello 3: errore per valore di τₘ ──────────────────────
ax = axes[2]
tau_true1 = np.array(r1["tau_true"])
tau_pred1 = np.array(r1["tau_pred"])
tau_true2 = np.array(r2["tau_true"])
tau_pred2 = np.array(r2["tau_pred"])

err1 = np.abs(tau_true1 - tau_pred1)
err2 = np.abs(tau_true2 - tau_pred2)

ax.bar(np.array(r1["tau_true"]) - 0.3, err1,
       width=0.5, color="steelblue", alpha=0.8, label="V1")
ax.bar(np.array(r2["tau_true"]) + 0.3, err2,
       width=0.5, color="darkorange", alpha=0.8, label="V2")
ax.set_xlabel("τₘ vero (s)")
ax.set_ylabel("|errore| (s)")
ax.set_title("Errore assoluto per τₘ")
ax.legend()

plt.tight_layout()
plt.savefig("dataset/comparison.png", dpi=150)
print("Salvato dataset/comparison.png")
