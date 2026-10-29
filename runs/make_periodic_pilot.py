"""Periodic-light pilot (2026-10-27): is tau_m readable from the response to each light-on?

Square wave starting in the dark, 40 s off / 40 s on, six cycles (480 s). After 40 s of dark
the box memory of every cell is reset to dark_tbr, so each light-on is a repeated step
response: cells are attracted up the gradient until their memory crosses the threshold,
then reverse. With our rule (average of the tumbling rate) and f_R saturated on the ring,
the crossing comes after p*tau with p* = (F_thr - 1.75)/(F_ring - 1.75) = 0.94-0.98, almost
independent of power; the 45 uW map checks that claim.

Map: the lab ring of session 103 (tbr_fields_lab/ill103, 180 uW) as a stand-in until the five
new rings are characterised, plus the same profile scaled to 45 uW (tbr_fields_lab/ill103_p45).
tau_m 5, 10, 20 s, 4 seeds, uniform start, frames every 0.5 s.
Logs were first written to the system disk (LaCie unmounted) and moved the same day to
LaCie PINN3/sims/pilota_ill103/ (~30 MB per run); analysis: /home/user/Scrivania/PINN3/periodic_pilot.py.
"""
import os, re, csv, sys
import numpy as np

BASE = "/home/user/Scrivania/pulsed_simulation"
RUNS = f"{BASE}/runs"
OUT = "/media/user/LaCie/PINN3/sims/pilota_ill103"
TEMPLATE = f"{RUNS}/lab103_tau8_on10_off0.01_seed0/parameters_multi.yaml"
DT = 1e-4
TAUS = (5, 10, 20)
SEEDS = range(1, 5)
ON, OFF, T_STOP, LOG = 40, 40, 480.0, 0.5
MAPS = {180: f"{BASE}/tbr_fields_lab/ill103", 45: f"{BASE}/tbr_fields_lab/ill103_p45"}

# 45 uW map: the session-103 radial profile scaled by 45/180, written exactly as lab_maps.py does
if not os.path.exists(f"{MAPS[45]}/tbr_map_ring_45muW.dat"):
    sys.path.insert(0, BASE)
    from lab_maps import write_maps
    E = np.loadtxt(f"{MAPS[180]}/radial_energy_density.txt")
    from pathlib import Path
    write_maps(E*45/180, Path(MAPS[45]), 45)
    np.savetxt(f"{MAPS[45]}/radial_energy_density.txt", E*45/180,
               header="uW/um^2 vs radius, 1.041667 um steps; ill103 profile scaled to 45 uW")


def yaml_for(power, tau, seed):
    y = open(TEMPLATE).read()
    subs = {"power_input": f"{power}  # see tbr_fields link",
            "time_mem": int(round(tau/DT)), "light_on_time": ON, "light_off_time": OFF,
            "LIGHT_START_OFF": 1, "pulse_start_time": 0.0, "NOCENTER": 0,
            "seed_offset": seed, "grad_sign_thresh": 3.0708352}
    for k, v in subs.items():
        y, n = re.subn(rf"^{k} : .*$", f"{k} : {v}", y, flags=re.M)
        assert n == 1, k
    assert not re.search(r"^(t_stop|log_interval|print_skip_exp) :", y, flags=re.M)
    return y + f"t_stop : {T_STOP}\nlog_interval : {LOG}\n"


runs = [(f"per{p}_tau{t}_on{ON}_off{OFF}_seed{s}", p, t, s) for p in MAPS for t in TAUS for s in SEEDS]
for r in runs:
    assert not os.path.exists(f"{RUNS}/{r[0]}"), r[0]

for name, power, tau, s in runs:
    rd, od = f"{RUNS}/{name}", f"{OUT}/{name}"
    os.makedirs(rd); os.makedirs(f"{od}/logs", exist_ok=True)
    y = yaml_for(power, tau, s)
    open(f"{rd}/parameters_multi.yaml", "w").write(y)
    open(f"{od}/parameters_multi.yaml", "w").write(y)
    os.symlink(f"{od}/logs", f"{rd}/logs")
    os.symlink(MAPS[power], f"{rd}/tbr_fields")

with open(f"{RUNS}/todo_periodic.txt", "w") as f:
    for r in sorted(runs, key=lambda r: -r[2]):     # longest memory first
        f.write(r[0] + "\n")
with open(f"{OUT}/manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "map", "power_uW", "tau_m_s", "light_on_s", "light_off_s", "seed"])
    for name, power, tau, s in runs:
        w.writerow([name, os.path.basename(MAPS[power]), power, tau, ON, OFF, s])
print(len(runs), "runs")
