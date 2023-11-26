# README

## the training script
run_pbt.sh is an example for how to training pbt model. Usage: sh run_pbt.sh <layout_name>

For customization, you only need to replace the 

```
nohup python pbt/pbt.py with fixed_mdp layout_name="unident_s" EX_NAME="pbt_unident_s" TOTAL_STEPS_PER_AGENT=1.1e7 REW_SHAPING_HORIZON=5e6 LR=8e-4 GPU_ID=3 POPULATION_SIZE=3 SEEDS="[$seed]" NUM_SELECTION_GAMES=6 VF_COEF=0.5 MINIBATCHES=10 TIMESTAMP_DIR=False > log/pbt_unident_s_$seed.txt &
```

and output info command:

```
echo "Train unident_s layout with seed: $seed using PBT"
```

or seeds list.

## how to use evaluation code

the evaluation code evaluation.py is the version from cole project, so it has more complex and unnecessary modules.

you just need to modify the layouts you need to test strating from line 156:

```
    pbt_model_paths = {
        # "simple": "pbt_simple",
        # "unident_s": "pbt_unident_s",
        "random1": "pbt_random1",
        # "random0": "pbt_random0",
        "random3": "pbt_random3",
    }
```

The results will be save at the ckpt folder.


