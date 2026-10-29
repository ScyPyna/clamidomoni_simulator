"""CSV export of the modulated-light dataset for the inverse-design work (2026-10-22).

  export_modulated.py one <run>   after a run: its trajectory file and its final positions
  export_modulated.py collect     after the batch: light_A.csv and final_positions.csv,
                                  over every run already exported

Layout in DATASET (see README.txt written by collect):
  light_A.csv                      IDsim,time,r_ext,A           one file, all simulations
  final_positions.csv              IDsim,IDalgae,x,y            one file, t = 120 s
  trajectories/traj_sim<ID>.csv    IDalgae,time,x,y             one file per simulation
Positions in um, origin at the ring centre, wrapped in the periodic box; time in s.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

RAW = "/media/user/LaCie/modulated_exps"
DATASET = "/media/user/LaCie/modulated_dataset"
PART_DIAM = 10.0            # log coordinates are in cell diameters
COLS = ["t", "id", "x", "y", "phi", "trate", "state", "light"]


def export_one(run):
    sim = int(run[3:7])
    log = glob.glob(f"{RAW}/{run}/logs/Simlog0_*.dat")[0]
    d = pd.read_csv(log, sep=" ", skiprows=2, header=None, names=COLS, usecols=["t", "id", "x", "y"])
    traj = pd.DataFrame({"IDalgae": d.id, "time": d.t.round(1),
                         "x": (d.x*PART_DIAM).round(2), "y": (d.y*PART_DIAM).round(2)})
    traj = traj.sort_values(["IDalgae", "time"], kind="stable")
    traj["time"] = traj.time.map("{:.1f}".format)                 # float_format would give 2 decimals
    os.makedirs(f"{DATASET}/trajectories", exist_ok=True)
    os.makedirs(f"{DATASET}/.final_parts", exist_ok=True)
    tmp = f"{DATASET}/trajectories/.traj_sim{sim:04d}.csv.tmp"
    traj.to_csv(tmp, index=False, float_format="%.2f")
    os.replace(tmp, f"{DATASET}/trajectories/traj_sim{sim:04d}.csv")   # a file present is a file complete
    final = traj[traj.time == f"{d.t.max():.1f}"].drop(columns="time")
    final.insert(0, "IDsim", sim)
    final.to_csv(f"{DATASET}/.final_parts/sim{sim:04d}.csv", index=False, float_format="%.2f")


def collect():
    man = pd.read_csv(f"{RAW}/manifest.csv")
    curves = np.load(f"{RAW}/curves.npy")
    t = np.load(f"{RAW}/curves_time.npy")
    done = sorted(int(os.path.basename(p)[3:7]) for p in glob.glob(f"{DATASET}/.final_parts/sim*.csv"))
    r_ext = man.set_index("id").r_out_um
    light = pd.DataFrame({"IDsim": np.repeat(done, len(t)), "time": np.tile([f"{x:.1f}" for x in t], len(done)),
                          "r_ext": np.repeat(r_ext.loc[done].to_numpy(), len(t)),
                          "A": curves[done].ravel()})
    light.to_csv(f"{DATASET}/light_A.csv", index=False, float_format="%.6g")
    final = pd.concat(pd.read_csv(f"{DATASET}/.final_parts/sim{i:04d}.csv") for i in done)
    final.to_csv(f"{DATASET}/final_positions.csv", index=False, float_format="%.2f")
    write_readme(len(done))
    print(f"{len(done)} simulations collected")


def write_readme(n):
    open(f"{DATASET}/README.md", "w").write(README.format(n=n, last=n - 1, rows_light=n*1201, rows_final=n*371))
    if os.path.exists(f"{DATASET}/README.txt"):    # superseded by the Markdown version
        os.remove(f"{DATASET}/README.txt")


README = """# Modulated-light dataset

Simulated *Chlamydomonas* cells (run-and-tumble with phototactic memory) under a ring of light
whose intensity changes smoothly in time. **{n} simulations**, 371 cells each, 120 s, one frame
every 0.1 s (t = 0.0 … 120.0, 1201 frames).

## Files

| File | Columns | Content |
|---|---|---|
| `light_A.csv` | `IDsim, time, r_ext, A` | light history of every simulation, one file ({rows_light:,} rows) |
| `final_positions.csv` | `IDsim, IDalgae, x, y` | cell positions at t = 120.0 s, one file ({rows_final:,} rows) |
| `trajectories/traj_sim<ID>.csv` | `IDalgae, time, x, y` | full trajectories, **one file per simulation** (`<ID>` = IDsim with 4 digits, e.g. `traj_sim0002.csv`); 445,571 rows, ≈10.4 MB each |

The trajectories are split per simulation because a single file would hold 2.2 billion rows
(≈52 GB) and could not be loaded in memory.

## Columns

| Column | Meaning | Units / values |
|---|---|---|
| `IDsim` | simulation identifier | 0 … {last} |
| `IDalgae` | cell identifier **within** a simulation | 0 … 370 |
| `time` | time since the start of the simulation (light on at t = 0, A = 1) | s, 0.0 … 120.0, step 0.1 |
| `r_ext` | outer radius of the light ring | µm: 245, 290, 335, 380, 420 |
| `A` | relative light intensity on the ring | 0 … 1 |
| `x`, `y` | cell position | µm |

## Conventions

- **Origin** at the centre of the ring.
- **Periodic box**: square, side 1207.3 µm, so x, y ∈ [−603.65, 603.65]. Positions are wrapped
  into the box: a cell leaving through one edge re-enters from the opposite one, which shows up
  as a jump of ≈1207 µm in a trajectory. To unwrap a trajectory, with L = 1207.3:
  `dx = np.diff(x); dx -= L * np.round(dx / L)`, then cumulative sum.
- **IDalgae** is only meaningful inside one simulation: cell 5 of two different simulations are
  unrelated.
- Rows of a trajectory file are sorted by `IDalgae`, then `time`.
- **1000 simulations per ring**: r_ext = 245, 290, 335, 380, 420 µm for IDsim % 5 = 0, 1, 2, 3, 4.

## Light

- Ring with inner radius 150 µm (all simulations) and outer radius `r_ext`; flat plateau with
  soft edges (erf profile, 10 µm wide); dark everywhere else.
- Intensity on the ring = **A(t) × 1.1·10⁻⁴ µW/µm²**.
- A(t) is smooth and random, starts at 1 and stays in [0, 1]. Every simulation has its own curve,
  with a correlation time between 5 and 30 s; after the first tens of seconds A is uniformly
  distributed over [0, 1].
- The simulator applies A exactly as listed in `light_A.csv`, updated every 0.1 s.

## Model

- Run-and-tumble swimming with phototactic memory: memory time τ_m = 10 s, swimming speed
  50 µm/s, cell diameter 10 µm, interactions between cells included.
- **Initial condition**: at t = 0 the cells sit on the **same square lattice** (spacing ≈ 60 µm,
  covering the whole box) in every simulation. Initial orientations and noise are random and
  differ between simulations (random seed = IDsim).
- Integration time step 10⁻⁴ s; frames saved every 0.1 s.

## Quick start

```python
import pandas as pd

light = pd.read_csv("light_A.csv")
final = pd.read_csv("final_positions.csv")
traj = pd.read_csv("trajectories/traj_sim0002.csv")

A2 = light[light.IDsim == 2]        # light history of simulation 2
end2 = final[final.IDsim == 2]      # its final state
```
"""

if __name__ == "__main__":
    if sys.argv[1] == "one":
        export_one(sys.argv[2])
    elif sys.argv[1] == "collect":
        collect()
