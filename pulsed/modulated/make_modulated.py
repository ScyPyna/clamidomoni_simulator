"""Modulated-light dataset (2026-10-22): 5000 runs, each with its own smooth random A(t).

The light on the ring is A(t) times the plateau of the map (tbr_fields_mod, built by mod_maps.py);
the simulator rebuilds the tumbling map f_R(A*I) every 0.1 s (MODULATED keys). Each run:
  - one of 5 rings, r_out = 245..420 um, assigned as id % 5 so any prefix of the queue covers all;
  - its own curve A(t) on 0..120 s every 0.1 s (1201 points), never reused;
  - tau_m 10 s, threshold 3.0708352, cells initialised uniformly (NOCENTER 0), seed_offset = id.

A(t): a Gaussian process g(t) with squared-exponential correlation of length tau_c (drawn
log-uniform in TAU_C), squashed by the normal CDF so that A is uniform on [0,1] at any time,
then blended from 1 at t = 0: A = 1 + (Phi(g) - 1) * (1 - exp(-(t/tau_c)^2)). It starts at 1, stays
in [0,1] and is smooth.

Outputs: run dirs in runs_mod/, logs on LaCie modulated_exps/<run>/logs, all curves in
modulated_exps/curves.npy (row = id) and the per-run table in modulated_exps/manifest.csv.
"""
import csv
import json
import os
import re

import numpy as np
from scipy.special import ndtr

BASE = "/home/user/Scrivania/pulsed_simulation"
RUNS = f"{BASE}/runs_mod"
OUT = "/media/user/LaCie/modulated_exps"
TEMPLATE = f"{BASE}/runs/tau8_on10_off5_seed0/parameters_multi.yaml"   # main pulsed grid
MAPS = json.load(open(f"{BASE}/tbr_fields_mod/summary.json"))
R_OUTS = (245, 290, 335, 380, 420)
N_RUNS = 5000
DT = 1e-4
T_STOP = 120.0
STEP = 0.1                  # curve resolution = map update = log interval
TAU_M = 10.0
TAU_C = (5.0, 30.0)         # correlation time of A(t), log-uniform
MASTER_SEED = 20261022

t = np.round(np.arange(0, T_STOP + STEP/2, STEP), 1)


def curve(i):
    rng = np.random.default_rng([MASTER_SEED, i])
    tau_c = float(np.exp(rng.uniform(*np.log(TAU_C))))
    s = tau_c/STEP/np.sqrt(2)                  # kernel width in samples: kernel * itself gives exp(-d^2/(2 tau_c^2))
    half = int(np.ceil(4*s))
    k = np.exp(-0.5*(np.arange(-half, half + 1)/s)**2)
    k /= np.sqrt((k**2).sum())                 # unit variance
    g = np.convolve(rng.standard_normal(len(t) + 2*half), k, mode="valid")
    A = 1 + (ndtr(g) - 1)*(1 - np.exp(-(t/tau_c)**2))
    return np.round(A, 6), tau_c


def yaml_for(i, power):
    y = open(TEMPLATE).read()
    subs = {"power_input": f"{power}  # flat ring, see tbr_fields link", "time_mem": int(round(TAU_M/DT)),
            "PULSED": 0, "NOCENTER": 0, "seed_offset": i, "Nlevel": 18, "grad_sign_thresh": 3.0708352}
    for k, v in subs.items():
        y, n = re.subn(rf"^{k} : .*$", f"{k} : {v}", y, flags=re.M)
        assert n == 1, k
    y += (f"\nMODULATED : 1 # intensity map multiplied by A(t) from modulation.txt"
          f"\nmodulation_id : {i}\nmodulation_update : {STEP}\nt_stop : {T_STOP}\nlog_interval : {STEP}\n")
    return y


runs = [(f"mod{i:04d}_rout{R_OUTS[i % 5]}", i, R_OUTS[i % 5]) for i in range(N_RUNS)]
for name, _, _ in runs:
    assert not os.path.exists(f"{RUNS}/{name}"), name
os.makedirs(OUT, exist_ok=True)

curves = np.empty((N_RUNS, len(t)))
rows = []
for name, i, r_out in runs:
    A, tau_c = curve(i)
    curves[i] = A
    power = MAPS["rings"][str(r_out)]["power_uW"]
    rd, od = f"{RUNS}/{name}", f"{OUT}/{name}"
    os.makedirs(rd)
    os.makedirs(f"{od}/logs", exist_ok=True)
    y = yaml_for(i, power)
    open(f"{rd}/parameters_multi.yaml", "w").write(y)
    open(f"{od}/parameters_multi.yaml", "w").write(y)
    np.savetxt(f"{rd}/modulation.txt", np.column_stack([t, A]), fmt="%.1f %.6f")
    os.symlink(f"{od}/logs", f"{rd}/logs")
    os.symlink(f"{BASE}/tbr_fields_mod/rout{r_out}", f"{rd}/tbr_fields")
    rows.append([name, i, r_out, power, round(tau_c, 3), i])

np.save(f"{OUT}/curves.npy", curves)
np.save(f"{OUT}/curves_time.npy", t)
with open(f"{OUT}/manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "id", "r_out_um", "power_uW", "tau_c_s", "seed_offset"])
    w.writerows(rows)
with open(f"{RUNS}/todo.txt", "w") as f:
    f.writelines(r[0] + "\n" for r in runs)
print(len(runs), "runs;", "A range", curves.min(), curves.max(), "; A(0) all 1:", bool((curves[:, 0] == 1).all()))
