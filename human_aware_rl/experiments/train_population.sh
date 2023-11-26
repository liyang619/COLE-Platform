# [3410 9323 3234 2942 6156] [9297 5863 6805 788 8275] [103 9796 5474 8748 5650] [8957 8612 6269 7824 9071] [8381 4280 9980 3652 9755] [6927 4848 8300 839 5303]

# for seed in 3410 9323 3234 2942 6156
#     do
#         nohup python ppo/ppo.py with EX_NAME="fcp_sp_simple" layout_name="simple" REW_SHAPING_HORIZON=2.5e6 LR=1e-3 PPO_RUN_TOT_TIMESTEPS=6e6 OTHER_AGENT_TYPE="sp" SEEDS="[$seed]" VF_COEF=1 TIMESTAMP_DIR=False > log/fcp_sp_unident_s_$seed.txt &
#         echo "Train simple layout with seed: $seed using SP"
#         sleep 2
#     done

for seed in 3410 9323 3234 2942 6156
    do
        nohup python ppo/ppo.py with EX_NAME="fcp_sp_random3" layout_name="random3" REW_SHAPING_HORIZON=2.5e6 LR=8e-4 PPO_RUN_TOT_TIMESTEPS=8e6 OTHER_AGENT_TYPE="sp" SEEDS="[$seed]" VF_COEF=0.5 TIMESTAMP_DIR=False > log/fcp_sp_random3_$seed.txt &
        echo "Train random3 layout with seed: $seed using SP"
        sleep 2
    done
# python ppo/ppo.py with EX_NAME="ppo_sp_unident_s" layout_name="unident_s" REW_SHAPING_HORIZON=2.5e6 PPO_RUN_TOT_TIMESTEPS=7e6 LR=1e-3 GPU_ID=3 OTHER_AGENT_TYPE="sp" SEEDS="[2229, 7649, 7225, 9807,  386]" VF_COEF=0.5 TIMESTAMP_DIR=False
# python ppo/ppo.py with EX_NAME="ppo_sp_random1" layout_name="random1" REW_SHAPING_HORIZON=3.5e6 PPO_RUN_TOT_TIMESTEPS=1e7 LR=6e-4 OTHER_AGENT_TYPE="sp" SEEDS="[2229, 7649, 7225, 9807,  386]" VF_COEF=0.5 TIMESTAMP_DIR=False
# python ppo/ppo.py with EX_NAME="ppo_sp_random0" layout_name="random0" REW_SHAPING_HORIZON=2.5e6 PPO_RUN_TOT_TIMESTEPS=7.5e6 LR=8e-4 OTHER_AGENT_TYPE="sp" SEEDS="[2229, 7649, 7225, 9807,  386]" VF_COEF=0.5 TIMESTAMP_DIR=False
# python ppo/ppo.py with EX_NAME="ppo_sp_random3" layout_name="random3" REW_SHAPING_HORIZON=2.5e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=8e-4 GPU_ID=3 OTHER_AGENT_TYPE="sp" SEEDS="[2229, 7649, 7225, 9807,  386]" VF_COEF=0.5 TIMESTAMP_DIR=False