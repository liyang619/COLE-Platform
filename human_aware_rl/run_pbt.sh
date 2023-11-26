#!/bin/sh
# $1 layout


if [ $1 = "unident_s" ]; then
    for seed in 8015 5608 581 4221 3554
        do
            nohup python pbt/pbt.py with fixed_mdp layout_name="unident_s" EX_NAME="pbt_unident_s" TOTAL_STEPS_PER_AGENT=1.1e7 REW_SHAPING_HORIZON=5e6 LR=8e-4 GPU_ID=3 POPULATION_SIZE=5 SEEDS="[$seed]" NUM_SELECTION_GAMES=6 VF_COEF=0.5 MINIBATCHES=10 TIMESTAMP_DIR=True > log/pbt_unident_s_$seed.txt &
            echo "Train unident_s layout with seed: $seed using PBT"
            sleep 2
        done
elif [ $1 = "random1" ]; then
    for seed in 8015 5608 581 4221 3554
        do
            nohup python pbt/pbt.py with fixed_mdp layout_name="random1" EX_NAME="pbt_random1" TOTAL_STEPS_PER_AGENT=5e6 REW_SHAPING_HORIZON=4e6 LR=8e-4 GPU_ID=1 POPULATION_SIZE=5 SEEDS="[$seed]" NUM_SELECTION_GAMES=6 VF_COEF=0.5 MINIBATCHES=10 TIMESTAMP_DIR=True > log/pbt_random1_$seed.txt &
            echo "Train random1 layout with seed: $seed using PBT"
            sleep 2
    done
elif [ $1 = "random3" ]; then
    for seed in 8015 5608 581 4221 3554
        do
            nohup python pbt/pbt.py with fixed_mdp layout_name="random3" EX_NAME="pbt_random3" TOTAL_STEPS_PER_AGENT=6e6 REW_SHAPING_HORIZON=4e6 LR=1e-3 GPU_ID=1 POPULATION_SIZE=5 SEEDS="[$seed]" NUM_SELECTION_GAMES=6 VF_COEF=0.5 MINIBATCHES=10 TIMESTAMP_DIR=True > log/pbt_random3_$seed.txt &
            echo "Train random3 layout with seed: $seed using PBT"
            sleep 2
    done
fi

