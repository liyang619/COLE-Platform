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

## Getting Started to Train Baselines of ZSC on OVERCOOKED2 

### How to install requirements?
Run
```
conda create -n cole python==3.7
conda activate cole
pip install -r requirements.txt
sh ./install.sh
```

If you find there are errors about sacred, you can modify the get_commit_if_possible function in dependencies.py of sacred as
```
def get_commit_if_possible(filename):
    return None, None, None
```

### How to run the training code?
In human_aware_rl, run
`SP`:
```
bash experiments/ppo_sp_experiments.sh
```

`PBT`:
```
bash experiments/pbt_experiments.sh
```

`FCP`:
```
bash experiments/train_population.sh
bash exoerunebts/fcp_train.sh
```

`MEP`:
```
bash experiments/mep_train.sh
bash experiments/mep_train_s2.sh
```



