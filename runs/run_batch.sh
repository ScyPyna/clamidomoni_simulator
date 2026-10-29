#!/bin/bash
# Batch of pulsed-illumination runs: tau_m in {8,16} s, light on in {1,5,10,15} s,
# light off in {0.01,0.1,1,5,10} s. Each run is single threaded, so all of them go in
# parallel. The working directories are local and tiny, logs// is a symlink to the LaCie
# drive, so nothing but M1/M2 is written on the (full) system disk.
RUNS=/home/user/Scrivania/pulsed_simulation/runs
OUT=/media/user/LaCie/pulsed_exps
BIN=/home/user/Scrivania/pulsed_simulation/clamidomoni

echo "batch started $(date -Iseconds)" > $OUT/batch_status.txt

for d in $RUNS/tau*; do
  name=$(basename $d)
  (
    cd $d || exit 1
    start=$(date +%s)
    $BIN > $OUT/$name/stdout.log 2>&1
    rc=$?
    stop=$(date +%s)
    cp -f M1.dat M2.dat $OUT/$name/ 2>/dev/null
    echo "$name rc=$rc elapsed=$((stop-start))s" >> $OUT/batch_status.txt
  ) &
done
wait

echo "batch finished $(date -Iseconds)" >> $OUT/batch_status.txt
echo "runs completed: $(grep -c 'rc=0' $OUT/batch_status.txt)/40" >> $OUT/batch_status.txt
