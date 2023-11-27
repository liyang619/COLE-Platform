# README
This branch contains these features:
- Human-Human Play.
- Design and playing with LLM Agent.

**Attention**: If you've finish the "How to setup" part of main branch. You don't need to redo that. However, **you need to rerun the npm build command**, as there are some changes from the main branch.**
# Getting Started

## 1. How to setup
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

## 3. How to run

To run a human-human play, run:
```shell
python overcookedgym/overcooked-flask/app_human.py
```

The game is on http://127.0.0.1:8088
The game will start when two browser windows are all ready.

To play with LLM agent, run:
```shell
python overcookedgym/overcooked-flask/app_llm.py
```
You may need to fill your open-ai api key in app_llm.py and modify the prompt.txt to use your own prompt.
For deep customize, please refer the code of LLMBot and LLMAgent in app_llm.py.

# License
[MIT License](LICENSE.md)

# Citation
Please cite
 ```
@article{li2023cooperative,
  title={Cooperative Open-ended Learning Framework for Zero-shot Coordination},
  author={Li, Yang and Zhang, Shao and Sun, Jichen and Du, Yali and Wen, Ying and Wang, Xinbing and Pan, Wei},
  journal={arXiv preprint arXiv:2302.04831},
  year={2023}
}

@article{lou2023pecan,
  title={PECAN: Leveraging Policy Ensemble for Context-Aware Zero-Shot Human-AI Coordination},
  author={Lou, Xingzhou and Guo, Jiaxian and Zhang, Junge and Wang, Jun and Huang, Kaiqi and Du, Yali},
  journal={arXiv preprint arXiv:2301.06387},
  year={2023}
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
