"""Batch 6 (2026-09-29): the lab protocols simulated with the lab's own illumination maps.

Each lab condition is run with the map of its acquisition session (tbr_fields_lab/ill<S>, built
by lab_maps.py) at the session power, for tau_m 4, 8, 16 s, the working threshold and 5 seeds.
Session 104 used exactly the illumination and power of 103, so its runs point at ill103.
Run names keep the session label; the run's tbr_fields symlink says which map it used.
"""
import os, re, csv

BASE = "/home/user/Scrivania/pulsed_simulation"
RUNS = f"{BASE}/runs"
OUT = "/media/user/LaCie/pulsed_exps"
TEMPLATE = f"{RUNS}/tau4_on20_off5_seed3/parameters_multi.yaml"   # main grid, threshold 3.0708352
DT = 1e-4
TAUS = (4, 8, 16)
SEEDS = range(5)
MAP_OF = {104: 103}
# (session, power uW, light on s, light off s), from the lab's acquisition table
CONDITIONS = [(98, 150, 1, 0.01),
              (99, 180, 1, 0.1), (99, 180, 1, 1), (99, 180, 5, 0.01),
              (100, 140, 5, 0.1), (100, 140, 5, 1),
              (101, 140, 5, 5), (102, 140, 5, 10),
              (103, 180, 10, 0.01),
              (104, 180, 10, 0.1), (104, 180, 10, 1), (104, 180, 10, 5), (104, 180, 10, 10),
              (105, 180, 15, 0.01), (105, 180, 15, 0.1), (105, 180, 15, 1), (105, 180, 15, 5), (105, 180, 15, 10)]


def yaml_for(power, tau, on, off, seed):
    y = open(TEMPLATE).read()
    subs = {"power_input": f"{power}  # lab session map, see tbr_fields link",
            "time_mem": int(round(tau/DT)), "light_on_time": on, "light_off_time": off,
            "seed_offset": seed, "grad_sign_thresh": 3.0708352}
    for k, v in subs.items():
        y, n = re.subn(rf"^{k} : .*$", f"{k} : {v}", y, flags=re.M)
        assert n == 1, k
    return y


runs = []
for sess, power, on, off in CONDITIONS:
    for tau in TAUS:
        for s in SEEDS:
            runs.append((f"lab{sess}_tau{tau}_on{on}_off{off:g}_seed{s}", sess, power, tau, on, off, s))
assert len({r[0] for r in runs}) == len(runs)
for r in runs:
    assert not os.path.exists(f"{RUNS}/{r[0]}"), r[0]

for name, sess, power, tau, on, off, s in runs:
    mapdir = f"{BASE}/tbr_fields_lab/ill{MAP_OF.get(sess, sess)}"
    assert os.path.exists(f"{mapdir}/tbr_map_ring_{power}muW.dat"), mapdir
    rd, od = f"{RUNS}/{name}", f"{OUT}/{name}"
    os.makedirs(rd); os.makedirs(f"{od}/logs", exist_ok=True)
    y = yaml_for(power, tau, on, off, s)
    open(f"{rd}/parameters_multi.yaml", "w").write(y)
    open(f"{od}/parameters_multi.yaml", "w").write(y)
    os.symlink(f"{od}/logs", f"{rd}/logs")
    os.symlink(mapdir, f"{rd}/tbr_fields")

duty = lambda r: r[4]/(r[4] + r[5])
with open(f"{RUNS}/todo6.txt", "w") as f:
    for r in sorted(runs, key=duty, reverse=True):
        f.write(r[0] + "\n")
with open(f"{OUT}/manifest_batch6.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "session", "map", "power_uW", "tau_m_s", "light_on_s", "light_off_s", "seed", "grad_sign_thresh"])
    for name, sess, power, tau, on, off, s in runs:
        w.writerow([name, sess, f"ill{MAP_OF.get(sess, sess)}", power, tau, on, off, s, 3.0708352])
print(len(runs), "runs")
