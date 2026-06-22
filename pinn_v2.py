import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

# ── device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── costanti fisiche ─────────────────────────────────────────
R_MAX     = 900.0   # µm
TAU_MAX   = 20.0    # s  - per normalizzare il target
N_BINS    = 50
N_TIME    = 2049

# ── dataset ──────────────────────────────────────────────────
class ChlamyDataset(Dataset):
    def __init__(self, samples, indices):
        self.samples = [samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        rho   = torch.tensor(s["rho"], dtype=torch.float32)
        T_r   = torch.tensor(s["T_r"], dtype=torch.float32)
        tau_m = torch.tensor(s["tau_m"] / TAU_MAX, dtype=torch.float32)
        return rho, T_r, tau_m


# ── architettura ─────────────────────────────────────────────
class ChlamyPINN(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder temporale leggero
        self.temporal_enc = nn.Sequential(
            nn.Conv1d(N_BINS, 32, kernel_size=32, stride=16),  # → (32, 126)
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4),         # → (64, 30)
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),                            # → (64, 4)
        )

        # fully connected minimale
        self.fc = nn.Sequential(
            nn.Flatten(),          # → 256
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, rho):
        x = self.temporal_enc(rho)
        x = self.fc(x)
        return x.squeeze(-1)

# ── loss fisica ───────────────────────────────────────────────
def physics_loss(rho, T_r, r_centers, dt, D=0.0247, v0=50.0):
    dr = float(r_centers[1] - r_centers[0])
    r  = torch.tensor(r_centers, dtype=torch.float32).to(rho.device)
    r  = r[None, :, None]  # (1, n_bins, 1)

    # ∂ρ/∂t — differenze finite nel tempo (dimensione 2)
    drho_dt = (rho[:, :, 2:] - rho[:, :, :-2]) / (2.0 * dt)
    # shape: (batch, n_bins, T-2)

    # lavoriamo su slice temporale centrale per allineare tutto
    rho_c   = rho[:, :, 1:-1]      # (batch, n_bins, T-2)
    T_r_c   = T_r[:, :, 1:-1]      # (batch, n_bins, T-2)

    # termine diffusivo: D * (1/r) * ∂_r(r * ∂_r ρ)
    # differenze finite in r (dimensione 1)
    drho_dr   = (rho_c[:, 2:, :] - rho_c[:, :-2, :]) / (2.0 * dr)
    # shape: (batch, n_bins-2, T-2)
    r_mid     = r[:, 1:-1, :]      # (1, n_bins-2, 1)
    r_drho_dr = r_mid * drho_dr    # (batch, n_bins-2, T-2)

    d_r_drho_dr = (r_drho_dr[:, 2:, :] - r_drho_dr[:, :-2, :]) / (2.0 * dr)
    # shape: (batch, n_bins-4, T-2)
    r_inner     = r[:, 2:-2, :]    # (1, n_bins-4, 1)
    diff_term   = D * d_r_drho_dr / r_inner.clamp(min=1.0)

    # termine advettivo: v₀ * (1/r) * ∂_r(r * T_r)
    rT_r        = r_mid * T_r_c[:, 1:-1, :]   # (batch, n_bins-2, T-2) → prima taglia i bordi
    rT_r        = r[:, 1:-1, :] * T_r_c[:, 1:-1, :]
    d_rT_r      = (rT_r[:, 2:, :] - rT_r[:, :-2, :]) / (2.0 * dr)
    # shape: (batch, n_bins-4, T-2) — stesso di diff_term ✓
    adv_term    = v0 * d_rT_r / r_inner.clamp(min=1.0)

    # residuo — stessa shape per tutti i termini ✓
    residuo = drho_dt[:, 2:-2, :] - diff_term + adv_term
    return torch.mean(residuo ** 2)

# ── training ─────────────────────────────────────────────────
def train():
    # carica dataset
    data = np.load("dataset/dataset.npy", allow_pickle=True)
    n    = len(data)
    print(f"Campioni totali: {n}")

    # split: 70/15/15
    idx   = np.random.permutation(n)
    n_tr  = int(0.7 * n)
    n_val = int(0.15 * n)
    idx_tr  = idx[:n_tr]
    idx_val = idx[n_tr:n_tr+n_val]
    idx_te  = idx[n_tr+n_val:]
    print(f"Train: {len(idx_tr)}  Val: {len(idx_val)}  Test: {len(idx_te)}")

    ds_tr  = ChlamyDataset(data, idx_tr)
    ds_val = ChlamyDataset(data, idx_val)
    ds_te  = ChlamyDataset(data, idx_te)

    dl_tr  = DataLoader(ds_tr,  batch_size=4, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=4)
    dl_te  = DataLoader(ds_te,  batch_size=4)

    # modello
    model = ChlamyPINN().to(device)
    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    mse       = nn.MSELoss()

    lambda_phys = 0.01   # peso del termine fisico

    best_val = float("inf")
    history  = {"train": [], "val": []}

    for epoch in range(300):
        # ── train ──
        model.train()
        loss_tr = 0.0
        for rho, T_r, tau in dl_tr:
            rho, T_r, tau = rho.to(device), T_r.to(device), tau.to(device)
            optimizer.zero_grad()
            tau_pred  = model(rho)
            loss_data = mse(tau_pred, tau)
            loss_phys = physics_loss(rho, T_r,
                                    data[0]["r_centers"],
                                    float(data[0]["t"][1] - data[0]["t"][0]))
            loss = loss_data + lambda_phys * loss_phys
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_tr += loss_data.item()
        loss_tr /= len(dl_tr)

        # ── validation ──
        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for rho, T_r, tau in dl_val:
                rho, T_r, tau = rho.to(device), T_r.to(device), tau.to(device)
                tau_pred  = model(rho)
                loss_val += mse(tau_pred, tau).item()

        loss_val /= max(len(dl_val), 1)

        scheduler.step()
        history["train"].append(loss_tr)
        history["val"].append(loss_val)

        if loss_val < best_val:
            best_val = loss_val
            torch.save(model.state_dict(), "dataset/best_model.pt")

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | loss_tr={loss_tr:.4f} | loss_val={loss_val:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    # ── test ──
    model.load_state_dict(torch.load("dataset/best_model.pt"))
    model.eval()
    tau_true_all, tau_pred_all = [], []
    with torch.no_grad():
        for rho, T_r, tau in dl_te:
            rho = rho.to(device)
            tau_pred = model(rho).cpu()
            tau_true_all.append(tau)
            tau_pred_all.append(tau_pred)

    tau_true = torch.cat(tau_true_all).numpy() * TAU_MAX
    tau_pred = torch.cat(tau_pred_all).numpy() * TAU_MAX
    print("\n── Risultati Test ──")
    for t, p in zip(tau_true, tau_pred):
        print(f"  τₘ vero: {t:.1f}s  |  τₘ predetto: {p:.2f}s  |  errore: {abs(t-p):.2f}s")
    rmse = np.sqrt(np.mean((tau_true - tau_pred)**2))
    print(f"\nRMSE: {rmse:.3f} s")

    # ── plot loss ──
    plt.figure(figsize=(10, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"],   label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dataset/training_history.png", dpi=120)
    print("Salvato dataset/training_history.png")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train()
