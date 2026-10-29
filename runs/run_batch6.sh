#!/bin/bash
# Batch 6: lab protocols with the lab illumination maps (see make_batch6.py).
# runs/todo6.txt holds the run names still to do, longest first so the tail of the
# queue is short. NJOBS stays below the core count to leave the machine usable.
RUNS=/home/user/Scrivania/pulsed_simulation/runs
OUT=/media/user/LaCie/pulsed_exps
BIN=/home/user/Scrivania/pulsed_simulation/clamidomoni
NJOBS=48

echo "batch started $(date -Iseconds) - $(wc -l < $RUNS/todo6.txt) runs" > $OUT/batch6_status.txt

run_one() {
  name=$1
  cd /home/user/Scrivania/pulsed_simulation/runs/$name || exit 1
  start=$(date +%s)
  /home/user/Scrivania/pulsed_simulation/clamidomoni > /media/user/LaCie/pulsed_exps/$name/stdout.log 2>&1
  rc=$?
  stop=$(date +%s)
  cp -f M1.dat M2.dat /media/user/LaCie/pulsed_exps/$name/ 2>/dev/null
  echo "$name rc=$rc elapsed=$((stop-start))s" >> /media/user/LaCie/pulsed_exps/batch6_status.txt
}
export -f run_one

echo $$ > $OUT/batch6_pid.txt
xargs -a $RUNS/todo6.txt -P $NJOBS -I{} bash -c 'run_one {}'

echo "batch finished $(date -Iseconds)" >> $OUT/batch6_status.txt
echo "ok: $(grep -c 'rc=0' $OUT/batch6_status.txt) / $(wc -l < $RUNS/todo6.txt)" >> $OUT/batch6_status.txt
