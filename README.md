# COLE: Cooperative Open-ended Learning Framework for Zero-shot Human-AI Coordination

Welcome to the COLE repository! The official GitHub repository is for our series work "Cooperative Open-ended Learning Framework for Zero-shot Coordination" (accepted by ICML2023) and "Tackling Cooperative Incompatibility for Zero-Shot Human-AI Coordination". 
You can access our hands-on [demo page](https://sites.google.com/view/cole-jair).
Below, you'll find a list of distinct features that our repository offers.

<span style="color:red;">Update: New Features are Available:</span> 
- 1. Support Human-Human Experiments (human branch)
- 2. Play with LLM Agent like GPT-4 (human branch) 
- 3. Training your own COLE_SV agent (cole_training branch) 
- 4. ZSC baseline agents including SP, FCP, PBT, MEP (baseline_training branch).


## Features

This repository presents a human-AI evaluation platform centered on the popular game Overcooked 2, created specifically to facilitate experiments involving human-AI interaction. Overcooked is an engaging, fully cooperative game that requires two players to work in concert. The architecture of the system is outlined below.

Here, you're granted the ability to:
- Upload your weights (main branch)
- Customize the human questionnaire (main branch)
- Configure game settings (main branch)
- Human-Human Play (human branch)
- Play with LLM agent like gpt-4 (human branch)
- Training your own COLE_SV agent (cole_training branch) and ZSC baseline agents including SP, FCP, PBT, MEP (baseline_training branch)
- And many more!

The usage of each ability, please refer to the corresponding branch.


This repository introduces a human-AI evaluation platform built around the Overcooked game, designed to support Human-AI experiments. Overcooked, a two-player fully cooperative game. The system is shown as follows.

<p align="center">
  <img src="./images/system_model.png" width="90%">
  <br>
</p>
Here, you're granted the ability to:

- Upload your weights (main branch)
- Customize the human questionnaire (main branch)
- Configure game settings (main branch)
- Play with LLM agent like gpt-4 (human branch)
- Training your own COLE_SV agent (cole branch) and ZSC baseline agents including SP, FCP, PBT, MEP (baseline branch)
- And many more!


## Getting Started

### 1. How to setup
Install [PantheonRL](https://github.com/Stanford-ILIAD/PantheonRL) in this repo
 ```shell
    conda create -n overcooked-vis python=3.7
    conda activate overcooked-vis
    pip install -r requirements.txt
    pip install -e .
```

Install mpi4py

```shell
conda install mpi4py
```

Install PyTorch (based on your CUDA version): https://pytorch.org/
(You don't actually need the GPU version to run the game)


Install human_aware_rl and its dependencies: overcooked_ai, baselines & stable_baselines
 ```shell
    cd overcookedgym/human_aware_rl
    pip install -e .
    cd overcooked_ai
    pip install -e .
    cd ..
    cd stable-baselines
    pip install -e .
    cd ..
    cd baselines
    pip install -e .
```

Here are instructions for building using npm.

You need to firstly install npm (if you can't do this, you can checkout our history version to get all built files)

**If you want to utilize human-human play or llm agents, please refer to `human` branch and remember that you need to re-run `npm run build` in `overcookedgym/overcooked-flask`**

In `overcookedgym/human_aware_rl/overcooked_ai/overcooked_ai_js`, run
```shell
npm install
sudo npm install browserify
```
Then
```shell
npm run build
npm run build-window
```
to build overcooked_ai game core.

Then, in `overcookedgym/overcooked-flask`
```shell
sed -i 's/overcook\"/overcooked\"/g' ../human_aware_rl/overcooked_ai/overcooked_ai_js/package.json
wget https://raw.githubusercontent.com/HumanCompatibleAI/overcooked_ai/37d14dd48ae93ad0363610a0a370221c47a79eb2/overcooked_ai_js/js/mdp.es6 -O ../human_aware_rl/overcooked_ai/overcooked_ai_js/js/mdp.es6
wget https://raw.githubusercontent.com/HumanCompatibleAI/overcooked_ai/37d14dd48ae93ad0363610a0a370221c47a79eb2/overcooked_ai_js/js/task.es6 -O ../human_aware_rl/overcooked_ai/overcooked_ai_js/js/task.es6
    
npm install
npm link ../human_aware_rl/overcooked_ai/overcooked_ai_js/

npm run build
```

### 2. How to load models

You need to put your model file in `./models`. You can get our trained models [here](https://drive.google.com/drive/folders/1s88a_muyG6pVlfcKDKop6R1Fhxr8dcGH?usp=share_link), including BC, self-play, population-based training, [FCP](https://arxiv.org/abs/2110.08176), [MEP](https://arxiv.org/abs/2112.11701), [COLE](https://arxiv.org/abs/2302.04831).

Also, you need to put this BC [data](https://drive.google.com/drive/folders/1gawHatRHkYor_J520uWgoQucqxwVPZwc?usp=drive_link) in `./data`.

**Note:** The layout names in code and google drive are not aligned with the layout names in those papers. Here is the mapping:

```code
PYTHON_LAYOUT_NAME_TO_ENV_NAME = {
    "unident_s": "Asymmetric Advantages",
    "simple": "Cramped Room",
    "random1": "Coordination Ring",
    "random0": "Forced Coordination",
    "random3": "Counter Circuit"
}
```

In addition, you can load your own models if they are trained using the [Human-Aware-RL](https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019) framework. 
Agents are loaded using the `get_agent_from_saved_model()` method, which loads tensorflow predictor models (`.pb` files), so you should save your agents in this style if you wish to load them into our framework. You can reference to the `save` method in `human_aware_rl/pbt/pbt.py` for saving agents that can be loaded.

To load your own models, you need to put them in the `./models` folder in a named folder (the folder names need to be the same for all layouts), and the models would be loaded upon starting the server. For example. If your algo is named `ABC`, then the folder structure should look like this:
```
-- models
  | --  simple
       | -- SP        <---- Baseline 1 
       | -- PBT       <---- Baseline 2
          ...
       | -- ABC       <---- Your Algorithm
  | --  unident_s
       | -- SP        <---- Baseline 1 
       | -- PBT       <---- Baseline 2
          ...
       | -- ABC       <---- Your Algorithm
  | --  random1
       | -- SP        <---- Baseline 1 
       | -- PBT       <---- Baseline 2
          ...
       | -- ABC       <---- Your Algorithm
  ...
``` 

### 3. How to run

```shell
python overcookedgym/overcooked-flask/app.py --trajs_savepath ./trajs --ckpts ./models
```

- `--ckpts`: Folder containing all the AI models to be loaded. Default is `./models`.
- `--port`: The port where you run the server process.
- `--trajs_savepath`: Optional trajectory save path, default is `./trajs`.
- `--questionnaire_savepath`: Optional questionnaire save path, default is `./questionnaire`.
- `--ip`: Default is LOCALHOST, we **recommend you replace it with your public network IP**, because of a known bug of Flask that may cause extreme lag when playing the game. The same applies when debugging, you should visit your machine's IP in your browser instead of LOCALHOST.

### 4. How to customize

#### Customize experiment statements
You can replace `configs/statement.md` by your experiment statement markdown file, then restarting your web process.

#### Customize before game questionnaire.
You can modify `configs/before_game.yaml` to customize your settings of before game questionnaire.

### 5. How to collect data
Questionnaire data are saved in `./questionnaire`, its corresponging co-play trajectorys is saved in `./trajs`.

We also privide a simple data processing scripts named `questionnaire_analyze.ipynb.`

# License
[MIT License](LICENSE.md)

# Citation
Please cite
 ```
@inproceedings{10.5555/3618408.3619252,
author = {Li, Yang and Zhang, Shao and Sun, Jichen and Du, Yali and Wen, Ying and Wang, Xinbing and Pan, Wei},
title = {Cooperative Open-Ended Learning Framework for Zero-Shot Coordination},
year = {2023},
publisher = {JMLR.org},
booktitle = {Proceedings of the 40th International Conference on Machine Learning},
articleno = {844},
numpages = {15},
location = {Honolulu, Hawaii, USA},
series = {ICML'23}
}
```
```
@misc{li2023tackling,
      title={Tackling Cooperative Incompatibility for Zero-Shot Human-AI Coordination}, 
      author={Yang Li and Shao Zhang and Jichen Sun and Wenhao Zhang and Yali Du and Ying Wen and Xinbing Wang and Wei Pan},
      year={2023},
      eprint={2306.03034},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
 ```
 ```
 @inproceedings{sarkar2022pantheonrl,
  title={PantheonRL: A MARL Library for Dynamic Training Interactions},
  author={Sarkar, Bidipta and Talati, Aditi and Shih, Andy and Sadigh, Dorsa},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={11},
  pages={13221--13223},
  year={2022}
}
 ```

 ```
@article{carroll2019utility,
  title={On the utility of learning about humans for human-ai coordination},
  author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
 ```
