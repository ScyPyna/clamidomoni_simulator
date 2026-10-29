#!/bin/bash
# Periodic-light pilot (see make_periodic_pilot.py). Logs on LaCie PINN3/sims/pilota_ill103/
# (run on 2026-10-27 to the system disk while LaCie was unmounted, then moved there).
RUNS=/home/user/Scrivania/pulsed_simulation/runs
OUT=/media/user/LaCie/PINN3/sims/pilota_ill103
NJOBS=24

echo "batch started $(date -Iseconds) - $(wc -l < $RUNS/todo_periodic.txt) runs" > $OUT/batch_status.txt

run_one() {
  name=$1
  OUT=/media/user/LaCie/PINN3/sims/pilota_ill103
  cd /home/user/Scrivania/pulsed_simulation/runs/$name || exit 1
  start=$(date +%s)
  /home/user/Scrivania/pulsed_simulation/clamidomoni > $OUT/$name/stdout.log 2>&1
  rc=$?
  stop=$(date +%s)
  echo "$name rc=$rc elapsed=$((stop-start))s" >> $OUT/batch_status.txt
}
export -f run_one

xargs -a $RUNS/todo_periodic.txt -P $NJOBS -I{} bash -c 'run_one {}'

echo "batch finished $(date -Iseconds)" >> $OUT/batch_status.txt
echo "ok: $(grep -c 'rc=0' $OUT/batch_status.txt) / $(wc -l < $RUNS/todo_periodic.txt)" >> $OUT/batch_status.txt
