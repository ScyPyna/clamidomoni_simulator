#!/bin/bash
# Modulated-light dataset: 5000 runs of 120 s (see make_modulated.py).
# runs_mod/todo.txt holds the run names; runs whose log already exists are skipped, so the
# script can be relaunched after an interruption. NJOBS stays below the core count to leave
# the machine usable; change it live with kill -USR1/-USR2 on the xargs pid.
RUNS=/home/user/Scrivania/pulsed_simulation/runs_mod
OUT=/media/user/LaCie/modulated_exps
NJOBS=${NJOBS:-48}

mountpoint -q /media/user/LaCie || { echo "LaCie not mounted"; exit 1; }
echo "batch started $(date -Iseconds) - $(wc -l < $RUNS/todo.txt) runs, $NJOBS jobs" >> $OUT/batch_status.txt

run_one() {
  name=$1
  out=/media/user/LaCie/modulated_exps/$name
  if ls $out/logs/Simlog0_*.dat >/dev/null 2>&1; then
    if [ ! -f /media/user/LaCie/modulated_dataset/trajectories/traj_sim${name:3:4}.csv ]; then
      /home/user/Scrivania/clamidomoni_simulator-main/venv_pinn/bin/python \
        /home/user/Scrivania/pulsed_simulation/runs/export_modulated.py one $name >> $out/stdout.log 2>&1
      echo "$name exported only (log existed) export=$?" >> /media/user/LaCie/modulated_exps/batch_status.txt
    else
      echo "$name skipped (log and csv exist)" >> /media/user/LaCie/modulated_exps/batch_status.txt
    fi
    return
  fi
  cd /home/user/Scrivania/pulsed_simulation/runs_mod/$name || exit 1
  start=$(date +%s)
  /home/user/Scrivania/pulsed_simulation/clamidomoni > $out/stdout.log 2>&1
  rc=$?
  stop=$(date +%s)
  cp -f M1.dat M2.dat $out/ 2>/dev/null
  # CSV for the students: this run's trajectories and final positions (export_modulated.py)
  if [ $rc -eq 0 ]; then
    /home/user/Scrivania/clamidomoni_simulator-main/venv_pinn/bin/python \
      /home/user/Scrivania/pulsed_simulation/runs/export_modulated.py one $name >> $out/stdout.log 2>&1
    xrc=$?
  else
    xrc=skip
  fi
  echo "$name rc=$rc elapsed=$((stop-start))s export=$xrc" >> /media/user/LaCie/modulated_exps/batch_status.txt
}
export -f run_one

echo $$ > $OUT/batch_pid.txt
xargs -a $RUNS/todo.txt -P $NJOBS -I{} bash -c 'run_one {}'

/home/user/Scrivania/clamidomoni_simulator-main/venv_pinn/bin/python \
  /home/user/Scrivania/pulsed_simulation/runs/export_modulated.py collect >> $OUT/batch_status.txt 2>&1
echo "batch finished $(date -Iseconds)" >> $OUT/batch_status.txt
echo "csv: $(ls /media/user/LaCie/modulated_dataset/trajectories/ | grep -c '^traj_sim.*csv$') / $(wc -l < $RUNS/todo.txt)" >> $OUT/batch_status.txt
