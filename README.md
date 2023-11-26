# COLE: Cooperative Open-ended Learning Framework for Zero-shot Human-AI Coordination

Welcome to the COLE repository! The official GitHub repository is for our series work "Cooperative Open-ended Learning Framework for Zero-shot Coordination" (accepted by ICML2023) and "Tackling Cooperative Incompatibility for Zero-Shot Human-AI Coordination". 
You can access our hands-on [demo page](https://sites.google.com/view/cole-jair).
Below, you'll find a list of distinct features that our repository offers.

<span style="color:red;">Update: New Features are Available:</span> 
- 1. Support Human-Human Experiments 
- 2. Play with LLM Agent like GPT-4 (main branch) 
- 3. Training your own COLE_SV agent (cole_training branch) 
- 4. ZSC baseline agents including SP, FCP, PBT, MEP (baseline_training branch).


## Features

This repository presents a human-AI evaluation platform centered on the popular game Overcooked 2, created specifically to facilitate experiments involving human-AI interaction. Overcooked is an engaging, fully cooperative game that requires two players to work in concert. The architecture of the system is outlined below.

Here, you're granted the ability to:
- Upload your weights (main branch)
- Customize the human questionnaire (main branch)
- Configure game settings (main branch)
- Human-Human Play (human branch)
- Play with LLM agent like gpt-4 (main branch)
- Training your own COLE_SV agent (cole_training branch) and ZSC baseline agents including SP, FCP, PBT, MEP (baseline_training branch)
- And many more!

## Getting Started to Train COLE_SV 

### How to install requirements?

``` bash
conda create -n cole python==3.7
conda activate cole
pip install -r requirements.txt
sh ./install.sh
```

### How to run the training code?
To start the main training process, the main training code is located at ./human_aware_rl/pbt/game_shapley_pbt.py. Therefore, you are required to navigate to the ./human_aware_rl directory and execute the subsequent scripts.

```bash
    python -u pbt/game_shapley_pbt.py with fixed_mdp init_pop_size=10 layout_name="random0" EX_NAME="pbt_random0" NUM_PBT_ITER=60 REW_SHAPING_HORIZON=60 LR=3e-3 GPU_ID=0 SEEDS="[SEED_LIST_YOU_LIKE_TO_USE]" VF_COEF=0.5 MINIBATCHES=10 PPO_RUN_TOT_TIMESTEPS=96000 TOTAL_BATCH_SIZE=48000 RATIO="RATIO_LIKE_1:3" 
    
    python -u pbt/game_shapley_pbt.py with fixed_mdp init_pop_size=10 layout_name="random1" EX_NAME="pbt_random1" NUM_PBT_ITER=60 REW_SHAPING_HORIZON=40 LR=8e-4 GPU_ID=0 SEEDS="[SEED_LIST_YOU_LIKE_TO_USE]" VF_COEF=0.5 MINIBATCHES=10 PPO_RUN_TOT_TIMESTEPS=96000 TOTAL_BATCH_SIZE=48000 RATIO="RATIO_LIKE_1:3" 
    
    python -u pbt/game_shapley_pbt.py with fixed_mdp init_pop_size=10 layout_name="random3" EX_NAME="pbt_random3" NUM_PBT_ITER=100 REW_SHAPING_HORIZON=60 LR=1e-3 GPU_ID=0 SEEDS="[SEED_LIST_YOU_LIKE_TO_USE]" VF_COEF=0.5 MINIBATCHES=10 PPO_RUN_TOT_TIMESTEPS=96000 TOTAL_BATCH_SIZE=48000 RATIO="RATIO_LIKE_1:3" 
    
    python -u pbt/game_shapley_pbt.py with fixed_mdp init_pop_size=10 layout_name="simple" EX_NAME="pbt_simple" NUM_PBT_ITER=80 REW_SHAPING_HORIZON=60 LR=2e-3 GPU_ID=0 SEEDS="[SEED_LIST_YOU_LIKE_TO_USE]" VF_COEF=0.5 MINIBATCHES=10 PPO_RUN_TOT_TIMESTEPS=96000 TOTAL_BATCH_SIZE=48000 RATIO="RATIO_LIKE_1:3" 
   
    python -u pbt/game_shapley_pbt.py with fixed_mdp init_pop_size=10 layout_name="unident_s" EX_NAME="pbt_unident_s" NUM_PBT_ITER=100 REW_SHAPING_HORIZON=75 LR=8e-4 GPU_ID=0 SEEDS="[SEED_LIST_YOU_LIKE_TO_USE]" VF_COEF=0.5 MINIBATCHES=10 PPO_RUN_TOT_TIMESTEPS=96000 TOTAL_BATCH_SIZE=48000 RATIO="RATIO_LIKE_1:3" 

```
In the scripts, you should replace the SEEDS and RATIO. Here are some example seeds for you:
```json
seeds = {
    "simple": [581, 4, 2],
    "unident_s": [5608,  4221, 1],
    "random1": [3, 5608, 3554],
    "random0": [8015, 2, 3554],
    "random3": [3, 5608,  581],
}
```

### How to monitor your training process?
Under main sub-folder human_aware_rl and run
```
python log2events.py
tensorboard --logdir ./data/events
```

### How to evaluate it?

To begin, it is essential that you acquire the Behavior Cloning weights by downloading them from the provided link [Google Drive](https://drive.google.com/drive/folders/1s88a_muyG6pVlfcKDKop6R1Fhxr8dcGH?usp=share_link). These weights should be placed in the following directory: ./human_aware_rl/data/bc_runs.
Subsequently, navigate to the human_aware_rl folder by executing cd human_aware_rl, and then run the get_best_model.py script with the following command:
```python
python test/get_best_model.py
```
This script is designed to identify and select the optimal weights based on the training payoff matrix.

Once you have the best model info (will be outputted to your DATA PATH), you can proceed to the evaluation phase by running the evaluation.py script:
```python
python test/evaluation.py
```
By default, this script evaluates all layouts using the best model information. If you require evaluation on a specific layout, you can append the --layout argument followed by the desired layout name, as in the command:
```python
python test/evaluation.py --layout "LAYOUT_NAME"
```
Replace "LAYOUT_NAME" with the actual name of the layout you wish to evaluate.

For cross-evaluation against other Zero-Shot Coordination (ZSC) methods, you may utilize the cross_evaluation.py script located at test/cross_evaluation.py. 


### How to visualize?

You need to create a new conda environment and run the visualization code.

``` bash
conda create -n model_conversion python=3.7

conda activate model_conversion

pip install scipy

pip install Cython

pip install numpy

python setup.py develop

pip install tensorflowjs

sh convert_model_to_web.sh "PATH_TO_SAVE" "LAYOUT_NAME" "SEED"
```
