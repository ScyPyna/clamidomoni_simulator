"""Batch 5 (2026-09-27): three extensions approved together.

  thr     threshold scan extended down to 2.2/2.4 for tau 8/16, plus tau 4 at 2.2/2.4/2.6
  onlong  light phases longer than the memory (ON 25/30/40 s) at the working threshold
  hires   logs every 0.1024 s (print_skip_exp 10) for P(theta) from velocities, as in the lab pipeline

Run directories are created in runs/, outputs on LaCie, exactly like the previous batches.
The queue (todo5.txt) puts thr and hires first, onlong last, each block longest-first.
"""
import os, re, csv

BASE = "/home/user/Scrivania/pulsed_simulation"
RUNS = f"{BASE}/runs"
OUT = "/media/user/LaCie/pulsed_exps"
TEMPLATE = f"{RUNS}/tau4_on20_off5_seed3/parameters_multi.yaml"  # main-grid yaml, threshold 3.0708352
WORK_THR = 3.0708352
SEEDS = range(5)
DT = 1e-4

def time_mem(tau):
    return int(round(tau / DT))

def yaml_for(tau, on, off, seed, thr, pse):
    y = open(TEMPLATE).read()
    y = re.sub(r"^time_mem : .*$", f"time_mem : {time_mem(tau)}", y, flags=re.M)
    y = re.sub(r"^grad_sign_thresh : .*$", f"grad_sign_thresh : {thr}", y, flags=re.M)
    y = re.sub(r"^light_on_time : .*$", f"light_on_time : {on}", y, flags=re.M)
    y = re.sub(r"^light_off_time : .*$", f"light_off_time : {off}", y, flags=re.M)
    y = re.sub(r"^seed_offset : .*$", f"seed_offset : {seed}", y, flags=re.M)
    if pse:
        y += f"print_skip_exp : {pse}\n"
    return y

runs = []  # (block, name, tau, on, off, seed, thr, pse)
for thr in (2.2, 2.4):
    for tau in (8, 16):
        for on in (10, 15, 20):
            for off in (1, 5, 10):
                for s in SEEDS:
                    runs.append(("thr", f"thr{thr:g}_tau{tau}_on{on}_off{off}_seed{s}", tau, on, off, s, thr, 0))
for thr in (2.2, 2.4, 2.6):
    for on in (10, 15, 20):
        for off in (1, 5, 10):
            for s in SEEDS:
                runs.append(("thr", f"thr{thr:g}_tau4_on{on}_off{off}_seed{s}", 4, on, off, s, thr, 0))
for tau in (4, 8, 16, 20):
    for on in (25, 30, 40):
        for off in (1, 2, 5, 10):
            for s in SEEDS:
                runs.append(("onlong", f"tau{tau}_on{on}_off{off}_seed{s}", tau, on, off, s, WORK_THR, 0))
for thr in (WORK_THR, 2.6):
    for tau in (8, 16):
        for on in (10, 15):
            for off in (1, 5):
                for s in SEEDS:
                    runs.append(("hires", f"hires_thr{thr:g}_tau{tau}_on{on}_off{off}_seed{s}", tau, on, off, s, thr, 10))

names = [r[1] for r in runs]
assert len(names) == len(set(names))
for r in runs:
    assert not os.path.exists(f"{RUNS}/{r[1]}"), f"{r[1]} already exists"

for block, name, tau, on, off, s, thr, pse in runs:
    rd, od = f"{RUNS}/{name}", f"{OUT}/{name}"
    os.makedirs(rd)
    os.makedirs(f"{od}/logs", exist_ok=True)
    y = yaml_for(tau, on, off, s, thr, pse)
    open(f"{rd}/parameters_multi.yaml", "w").write(y)
    open(f"{od}/parameters_multi.yaml", "w").write(y)
    os.symlink(f"{od}/logs", f"{rd}/logs")
    os.symlink(f"{BASE}/tbr_fields", f"{rd}/tbr_fields")

def duty(r):
    return r[3] / (r[3] + r[4])

first = sorted([r for r in runs if r[0] != "onlong"], key=duty, reverse=True)
last = sorted([r for r in runs if r[0] == "onlong"], key=duty, reverse=True)
with open(f"{RUNS}/todo5.txt", "w") as f:
    for r in first + last:
        f.write(r[1] + "\n")

with open(f"{OUT}/manifest_batch5.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "block", "tau_m_s", "time_mem_steps", "light_on_s", "light_off_s", "seed",
                "grad_sign_thresh", "print_skip_exp", "duty_cycle", "period_s"])
    for block, name, tau, on, off, s, thr, pse in runs:
        w.writerow([name, block, tau, time_mem(tau), on, off, s, thr, pse, round(on / (on + off), 4), on + off])

for b in ("thr", "onlong", "hires"):
    print(b, sum(r[0] == b for r in runs))
print("total", len(runs))
