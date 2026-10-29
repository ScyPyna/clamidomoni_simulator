#!/bin/bash
# Batch 5: threshold scan 2.2-2.6 (incl. tau 4), ON 25-40 s, high-res logs (see make_batch5.py).
# runs/todo5.txt holds the run names still to do, longest first so the tail of the
# queue is short. NJOBS stays below the core count to leave the machine usable.
RUNS=/home/user/Scrivania/pulsed_simulation/runs
OUT=/media/user/LaCie/pulsed_exps
BIN=/home/user/Scrivania/pulsed_simulation/clamidomoni
NJOBS=48

echo "batch started $(date -Iseconds) - $(wc -l < $RUNS/todo5.txt) runs" > $OUT/batch5_status.txt

run_one() {
  name=$1
  cd /home/user/Scrivania/pulsed_simulation/runs/$name || exit 1
  start=$(date +%s)
  /home/user/Scrivania/pulsed_simulation/clamidomoni > /media/user/LaCie/pulsed_exps/$name/stdout.log 2>&1
  rc=$?
  stop=$(date +%s)
  cp -f M1.dat M2.dat /media/user/LaCie/pulsed_exps/$name/ 2>/dev/null
  echo "$name rc=$rc elapsed=$((stop-start))s" >> /media/user/LaCie/pulsed_exps/batch5_status.txt
}
export -f run_one

echo $$ > $OUT/batch5_pid.txt
xargs -a $RUNS/todo5.txt -P $NJOBS -I{} bash -c 'run_one {}'

echo "batch finished $(date -Iseconds)" >> $OUT/batch5_status.txt
echo "ok: $(grep -c 'rc=0' $OUT/batch5_status.txt) / $(wc -l < $RUNS/todo5.txt)" >> $OUT/batch5_status.txt
