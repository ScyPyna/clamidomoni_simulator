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
        # rho: (N_BINS, N_TIME) → tensor float
        rho   = torch.tensor(s["rho"],   dtype=torch.float32)
        tau_m = torch.tensor(s["tau_m"] / TAU_MAX, dtype=torch.float32)
        return rho, tau_m


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
def physics_loss(rho_pred, rho_true, r_centers, t_values):
    """
    Penalizza violazioni dell'equazione di continuità:
    ∂ρ/∂t + ∇·J = 0
    Approssimata come: la densità deve variare smoothly nel tempo
    e lo shift radiale deve essere consistente con v₀
    """
    # differenze finite nel tempo
    drho_dt = rho_pred[:, :, 1:] - rho_pred[:, :, :-1]  # (batch, bins, T-1)
    
    # penalizza variazioni troppo brusche (regularizzazione fisica)
    smoothness = torch.mean(drho_dt ** 2)
    
    # penalizza asimmetrie non fisiche nella distribuzione radiale
    # ρ deve essere monotona decrescente lontano dal picco
    drho_dr = rho_pred[:, 1:, :] - rho_pred[:, :-1, :]  # (batch, bins-1, T)
    
    return smoothness + 0.1 * torch.mean(torch.relu(-drho_dr[:, -10:, :]) ** 2)


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
        for rho, tau in dl_tr:
            rho, tau = rho.to(device), tau.to(device)
            optimizer.zero_grad()
            tau_pred = model(rho)
            loss_data = mse(tau_pred, tau)
            loss      = loss_data
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_tr += loss.item()
        loss_tr /= len(dl_tr)

        # ── validation ──
        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for rho, tau in dl_val:
                rho, tau = rho.to(device), tau.to(device)
                tau_pred = model(rho)
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
        for rho, tau in dl_te:
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
