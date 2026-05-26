#!/bin/bash

mkdir -p logs

TIME_MEM_VALUES=(0 10000 30000 50000 80000 120000 160000 200000)
SEEDS=(42 123 777)

sim_counter=0

for tmem in "${TIME_MEM_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Lanciando time_mem=$tmem seed=$seed (sim $sim_counter)"
        
        cp parameters_multi_backup.yaml parameters_multi.yaml
        sed -i "s/^time_mem.*/time_mem : $tmem/" parameters_multi.yaml
        sed -i "s/^Nlevel.*/Nlevel : 18/" parameters_multi.yaml
        sed -i "s/^INDEPENDENT.*/INDEPENDENT : 1/" parameters_multi.yaml
        sed -i "s/^rho .*/rho : 0.001/" parameters_multi.yaml
        sed -i "s/^NperCore.*/NperCore : 1/" parameters_multi.yaml
        
        ./clamidomoni
        
        # rinomina usando il counter come identificatore unico
        for f in logs/Simlog0_*.dat; do
            [[ -f "$f" ]] && mv "$f" "logs/traj_tmem${tmem}_seed${seed}.dat"
        done
        for f in logs/Simqts0_*.dat; do
            [[ -f "$f" ]] && mv "$f" "logs/qts_tmem${tmem}_seed${seed}.dat"
        done
        
        sim_counter=$((sim_counter + 1))
    done
done

echo "Completate $sim_counter simulazioni"
