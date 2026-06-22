import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ── device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── costanti ─────────────────────────────────────────────────
TAU_MAX = 20.0
N_BINS  = 50
N_TIME  = 2049
D       = 0.0247   # µm²/s — coefficiente di diffusione traslazionale
v0      = 50.0     # µm/s  — velocità di nuoto


# ── dataset supervisionato (sintetico) ───────────────────────
class SimDataset(Dataset):
    def __init__(self, samples, indices):
        self.samples = [samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        rho   = torch.tensor(s["rho"],   dtype=torch.float32)
        T_r   = torch.tensor(s["T_r"],   dtype=torch.float32)
        tau_m = torch.tensor(s["tau_m"] / TAU_MAX, dtype=torch.float32)
        r_out = torch.tensor(s.get("r_out", 420.0) / 420.0, dtype=torch.float32)
        return rho, T_r, r_out, tau_m


# ── dataset non supervisionato (reale) ───────────────────────
class RealDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        rho   = torch.tensor(s["rho"], dtype=torch.float32)
        T_r   = torch.tensor(s["T_r"], dtype=torch.float32)
        r_out = torch.tensor(s.get("r_out", 420.0) / 420.0, dtype=torch.float32)
        return rho, T_r, r_out


# ── architettura ─────────────────────────────────────────────
class ChlamyPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_enc = nn.Sequential(
            nn.Conv1d(N_BINS, 32, kernel_size=32, stride=16),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(257, 64),   # 256 + 1 (r_out)
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, rho, r_out):
        x = self.temporal_enc(rho)
        x = torch.cat([x.flatten(1), r_out.unsqueeze(1)], dim=1)
        return self.fc(x).squeeze(-1)


# ── loss fisica — equazione di continuità in geometria radiale ──
def physics_loss(rho, T_r, r_centers, dt):
    dr  = float(r_centers[1] - r_centers[0])
    r   = torch.tensor(r_centers, dtype=torch.float32).to(rho.device)
    r   = r[None, :, None]

    # ∂ρ/∂t — shape: (batch, N_BINS, T-2)
    drho_dt = (rho[:, :, 2:] - rho[:, :, :-2]) / (2.0 * dt)

    # slice temporale centrale — shape: (batch, N_BINS, T-2)
    rho_c = rho[:, :, 1:-1]
    T_r_c = T_r[:, :, 1:-1]

    # termine diffusivo
    # ∂_r ρ — shape: (batch, N_BINS-2, T-2)
    drho_dr   = (rho_c[:, 2:, :] - rho_c[:, :-2, :]) / (2.0 * dr)
    r_mid     = r[:, 1:-1, :]                          # (1, N_BINS-2, 1)
    r_drho_dr = r_mid * drho_dr                        # (batch, N_BINS-2, T-2)
    # ∂_r(r*∂_r ρ) — shape: (batch, N_BINS-4, T-2)
    d_r_drho  = (r_drho_dr[:, 2:, :] - r_drho_dr[:, :-2, :]) / (2.0 * dr)
    r_inner   = r[:, 2:-2, :]                          # (1, N_BINS-4, 1)
    diff_term = D * d_r_drho / r_inner.clamp(min=1.0)  # (batch, N_BINS-4, T-2)

    # termine advettivo
    # r*T_r — shape: (batch, N_BINS-2, T-2)
    rT_r     = r_mid * T_r_c[:, 1:-1, :]
    # ∂_r(r*T_r) — shape: (batch, N_BINS-4, T-2)
    d_rT_r   = (rT_r[:, 2:, :] - rT_r[:, :-2, :]) / (2.0 * dr)
    adv_term = v0 * d_rT_r / r_inner.clamp(min=1.0)   # (batch, N_BINS-4, T-2)

    # residuo — allinea la dimensione temporale di drho_dt
    # drho_dt: (batch, N_BINS, T-2) → slice ai bin interni: (batch, N_BINS-4, T-2)
    residuo = drho_dt[:, 2:-2, :] - diff_term + adv_term
    return torch.mean(residuo ** 2)


# ── training ─────────────────────────────────────────────────
def train():
    # carica dataset sintetico
    sim_data = np.load("dataset/dataset_all.npy", allow_pickle=True)
    n        = len(sim_data)
    idx      = np.random.permutation(n)
    n_tr     = int(0.7 * n)
    n_val    = int(0.15 * n)
    idx_tr   = idx[:n_tr]
    idx_val  = idx[n_tr:n_tr + n_val]
    idx_te   = idx[n_tr + n_val:]

    ds_tr  = SimDataset(sim_data, idx_tr)
    ds_val = SimDataset(sim_data, idx_val)
    ds_te  = SimDataset(sim_data, idx_te)
    dl_tr  = DataLoader(ds_tr,  batch_size=8, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=8)
    dl_te  = DataLoader(ds_te,  batch_size=8)

    # carica dataset reale (non supervisionato)
    real_data = np.load("dataset/dataset_real.npy", allow_pickle=True)
    ds_real   = RealDataset(real_data)
    dl_real   = DataLoader(ds_real, batch_size=4, shuffle=True)

    print(f"Sintetico: {len(sim_data)} campioni  |  Reale: {len(real_data)} campioni")
    print(f"Train: {len(idx_tr)}  Val: {len(idx_val)}  Test: {len(idx_te)}")

    # modello inizializzato dai pesi pre-addestrati
    model = ChlamyPINN().to(device)
    model.load_state_dict(torch.load("dataset/best_model_all.pt",
                                      map_location=device))
    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}")
    print("Pesi inizializzati da best_model_all.pt")

    optimizer   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    mse         = nn.MSELoss()
    lambda_phys = 0.1   # peso del termine fisico

    # r_centers e dt per la loss fisica
    r_centers = sim_data[0]["r_centers"].astype(np.float32)
    dt        = 1.0 / 10.0  # fps = 10

    best_val = float("inf")
    history  = {"train": [], "val": [], "phys": []}

    real_iter = iter(dl_real)

    for epoch in range(300):
        # ── train ──────────────────────────────────────────
        model.train()
        loss_tr_sum   = 0.0
        loss_phys_sum = 0.0

        for rho, T_r, r_out, tau in dl_tr:
            rho, T_r, r_out, tau = (rho.to(device), T_r.to(device),
                                     r_out.to(device), tau.to(device))
            optimizer.zero_grad()

            # loss supervisionata sui dati sintetici
            tau_pred = model(rho, r_out)
            loss_sup = mse(tau_pred, tau)

            # loss fisica sui dati reali (non supervisionata)
            try:
                rho_r, T_r_r, r_out_r = next(real_iter)
            except StopIteration:
                real_iter = iter(dl_real)
                rho_r, T_r_r, r_out_r = next(real_iter)

            rho_r = rho_r.to(device)
            T_r_r = T_r_r.to(device)
            loss_ph = physics_loss(rho_r, T_r_r, r_centers, dt)

            loss = loss_sup + lambda_phys * loss_ph
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_tr_sum   += loss_sup.item()
            loss_phys_sum += loss_ph.item()

        loss_tr_sum   /= len(dl_tr)
        loss_phys_sum /= len(dl_tr)

        # ── validation ─────────────────────────────────────
        model.eval()
        loss_val_sum = 0.0
        with torch.no_grad():
            for rho, T_r, r_out, tau in dl_val:
                rho, T_r, r_out, tau = (rho.to(device), T_r.to(device),
                                         r_out.to(device), tau.to(device))
                tau_pred      = model(rho, r_out)
                loss_val_sum += mse(tau_pred, tau).item()
        loss_val_sum /= max(len(dl_val), 1)

        scheduler.step()
        history["train"].append(loss_tr_sum)
        history["val"].append(loss_val_sum)
        history["phys"].append(loss_phys_sum)

        if loss_val_sum < best_val:
            best_val = loss_val_sum
            torch.save(model.state_dict(), "dataset/best_model_domain.pt")

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | sup={loss_tr_sum:.5f} | "
                  f"phys={loss_phys_sum:.5f} | val={loss_val_sum:.5f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    # ── test ───────────────────────────────────────────────
    model.load_state_dict(torch.load("dataset/best_model_domain.pt"))
    model.eval()
    tau_true_all, tau_pred_all = [], []
    with torch.no_grad():
        for rho, T_r, r_out, tau in dl_te:
            rho, r_out = rho.to(device), r_out.to(device)
            tau_pred   = model(rho, r_out).cpu()
            tau_true_all.append(tau)
            tau_pred_all.append(tau_pred)

    tau_true = torch.cat(tau_true_all).numpy() * TAU_MAX
    tau_pred = torch.cat(tau_pred_all).numpy() * TAU_MAX
    print("\n── Risultati Test ──")
    for t, p in zip(tau_true, tau_pred):
        print(f"  τₘ vero: {t:.1f}s  |  predetto: {p:.2f}s  |  errore: {abs(t-p):.2f}s")
    rmse = np.sqrt(np.mean((tau_true - tau_pred) ** 2))
    print(f"\nRMSE: {rmse:.3f} s")

    # ── plot ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train"], label="train sup",  color="steelblue")
    axes[0].plot(history["val"],   label="val sup",    color="steelblue", ls="--")
    axes[0].set_title("Supervised loss (sintetico)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].plot(history["phys"], color="darkorange", label="physics loss (reale)")
    axes[1].set_title("Physics loss (dati reali non supervisionati)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("residuo equazione continuità")
    axes[1].legend()
    axes[1].set_yscale("log")

    tau_range = [0, 21]
    axes[2].plot(tau_range, tau_range, "k--", alpha=0.5, label="perfetto")
    axes[2].scatter(tau_true, tau_pred, s=60, zorder=5, color="steelblue")
    axes[2].set_xlabel("τₘ vero (s)")
    axes[2].set_ylabel("τₘ predetto (s)")
    axes[2].set_title(f"Test set — RMSE = {rmse:.3f} s")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("dataset/training_history_domain.png", dpi=120)
    print("Salvato dataset/training_history_domain.png")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train()
