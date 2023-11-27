import argparse
import copy
import json
import os
import time
import yaml
import random
from markdown import markdown

import numpy as np
from flask import Flask, jsonify, request

from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedGridworld, OvercookedState, PlayerState
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.utils import load_pickle
from overcookedgym.overcooked_utils import NAME_TRANSLATION
from pantheonrl.common.trajsaver import SimultaneousTransitions
from overcooked_ai_py.mdp.actions import Direction, Action
from pantheonrl.tf_utils import get_agent_from_saved_model
from flask_cors import CORS

import openai

# Overcooked action translation,
# the numbers are used internally to represent different actions
ACTION_ID_TRANSLATION = {0: "up", 1: "down", 2: "right", 3: "left", 4: "stay", 5: "interact"}
NAME_TRANSLATION_REVERSE = {
    v: k for k, v in NAME_TRANSLATION.items()
}
# Behavioral Cloning agents return 2d coords (or "interact") as action,
# so we need a translation table
BC_ACTION_TRANSLATION = {
    (0, 0): 4,  # stay
    "interact": 5,  # interact
    (0, -1): 0,  # up
    (-1, 0): 3,  # left
    (0, 1): 1,  # down
    (1, 0): 2,  # right
}


ACTION_TRANSLATION = {
    "[0, -1]": 0,
    "[0, 1]": 1,
    "[1, 0]": 2,
    "[-1, 0]":3,
    "[0, 0]": 4,
    "INTERACT": 5
}

POLICY_P0, POLICY_P1 = None, None
ALGO_P0, ALGO_P1 = None, None

MAX_AGENTS = 2


# AGENTS_TIMESTEP = [-1] * (2 * MAX_AGENTS)
AGENTS_TIMESTEP = np.ones(2*MAX_AGENTS) * (-1)

# AGENTS_ACTION = [[4] * 500] * (2 * MAX_AGENTS)
AGENTS_ACTION = np.ones((2*MAX_AGENTS, 500)) * 4


# Different game settings. Each pair means both player types, (player0, player1).
# "human" means the player is HUMAN KEYBOARD INPUT.
# "COLE" or "COLE_NOSP" are algorithms (Put models with same folder name into models/)
# AGENT_SETTINGS = {"0": ("human", "COLE"), "1": ("human", "COLE_NOSP"), "2": ("COLE", "COLE_NOSP")}
ALGO = "COLE"
ALGO_BASELINES = ["SP", "PBT", "MEP", "FCP"]
cur_algo_idx = -2
# HUMAN_LIST = ["human"]

ALL_LAYOUTS = ['simple', 'unident_s', 'random1', 'random0', 'random3']

AGENTS = {
    layout: dict() for layout in ALL_LAYOUTS
}
# LOCAL_IP = "118.195.241.173"
app = Flask(__name__)

CORS(app)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--layout_name", type=str, required=True, help="layout name")
# parser.add_argument('--ego', help="path for ego agent")
# parser.add_argument('--alt', help="path for alt agent")
parser.add_argument("--ckpts", type=str, default='./models', help="path of agent checkpoints")
parser.add_argument("--default", type=str, default= 0, help="default agent setting type")
parser.add_argument("--port", type=int, default=8088, help="port to run flask")
parser.add_argument("--seed", type=int, default=1, help="seed for model")
parser.add_argument("--dummy", type=bool, default=False, help="demo dummy partner. Won't move.")
parser.add_argument("--trajs_savepath", type=str, default=None, help="optional trajectory save path")
parser.add_argument(
    "--questionnaire_savepath", type=str, default="./questionnaire", help="optional questionnaire save path"
)

parser.add_argument("--sim_threads", type=int, default=30, help="simulation threads for trained agent")
parser.add_argument("--ip", type=str, default='192.168.1.114', help="your public network ip, default is localhost")
global ARGS
ARGS = parser.parse_args()
LOCAL_IP = ARGS.ip


def get_prediction(s, policy, layout_name, algo):
    """
    get agent action from observed state s and policy
    """
    if ARGS.dummy:
        return int(0)
    s = np.reshape(s, (1, s.shape[0], s.shape[1], s.shape[2]))
    action = policy(s)
    return int(action)


def process_state(state_dict, layout_name):
    def object_from_dict(object_dict):
        return ObjectState(**object_dict)

    def player_from_dict(player_dict):
        held_obj = player_dict.get("held_object")
        if held_obj is not None:
            player_dict["held_object"] = object_from_dict(held_obj)
        return PlayerState(**player_dict)

    def state_from_dict(state_dict):
        state_dict["players"] = [player_from_dict(p) for p in state_dict["players"]]
        object_list = [object_from_dict(o) for _, o in state_dict["objects"].items()]
        state_dict["objects"] = {ob.position: ob for ob in object_list}
        state_dict["all_orders"] = state_dict["order_list"]
        del state_dict["order_list"]
        return OvercookedState(**state_dict)

    state = state_from_dict(copy.deepcopy(state_dict))
    obs0, obs1 = MDPS[layout_name].lossless_state_encoding(state)
    return state


def convert_traj_to_simultaneous_transitions(traj_dict, layout_name):
    ego_obs = []
    alt_obs = []
    ego_act = []
    alt_act = []
    flags = []

    for state_list in traj_dict["ep_states"]:  # loop over episodes
        ego_obs.append([process_state(state, layout_name)[0] for state in state_list])
        alt_obs.append([process_state(state, layout_name)[1] for state in state_list])

        # check pantheonrl/common/wrappers.py for flag values
        flag = [0 for state in state_list]
        flag[-1] = 1
        flags.append(flag)

    for action_list in traj_dict["ep_actions"]:  # loop over episodes
        ego_act.append([joint_action[0] for joint_action in action_list])
        alt_act.append([joint_action[1] for joint_action in action_list])

    ego_obs = np.concatenate(ego_obs, axis=-1)
    alt_obs = np.concatenate(alt_obs, axis=-1)
    ego_act = np.concatenate(ego_act, axis=-1)
    alt_act = np.concatenate(alt_act, axis=-1)
    flags = np.concatenate(flags, axis=-1)

    return SimultaneousTransitions(
        ego_obs,
        ego_act,
        alt_obs,
        alt_act,
        flags,
    )


class LLMAgent:

    def __init__(self, mlp, agent_index) -> None:
        self.mlp = mlp
        self.agent_index = agent_index
        self.current_ml_action = None


    def find_motion_goals(self, state):

        am = self.mlp.ml_action_manager
        mdp = self.mlp.mdp
        motion_goals = []
        player = state.players[self.agent_index]
        pot_states_dict = mdp.get_pot_states(state)
        counter_objects = mdp.get_counter_objects_dict(
            state, list(mdp.terrain_pos_dict["X"])
        )
        if "pickup(onion)" in self.current_ml_action or "pickup_onion" in self.current_ml_action:
            motion_goals = am.pickup_onion_actions(state, counter_objects)
        elif "pickup(dish)" in self.current_ml_action or "pickup_dish" in self.current_ml_action:
            motion_goals = am.pickup_dish_actions(state, counter_objects)
        elif "put_onion_in_pot" in self.current_ml_action:
            motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
        elif "place_obj_on_counter" in self.current_ml_action:
            motion_goals = am.place_obj_on_counter_actions(state)
        elif "fill_dish_with_soup" in self.current_ml_action:
            motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
        elif "deliver_soup" in self.current_ml_action:
            motion_goals = am.deliver_soup_actions()
        elif "wait" in self.current_ml_action:
            motion_goals = am.wait_actions(player)
        else:
            # motion_goals = am.wait_actions(player)
            raise ValueError("Invalid action: {}".format(self.current_ml_action))


        return motion_goals
    

    def choose_motion_goal(self, start_pos_and_or, motion_goals):
        """Returns chosen motion goal (either boltzmann rationally or rationally), and corresponding action"""
        chosen_goal, chosen_goal_action = self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
        
        return chosen_goal, chosen_goal_action
    
    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """Returns action and goal that correspond to the cheapest plan among possible motion goals"""
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action


    def action(self, state):

        start_pos_and_or = state.players_pos_and_or[self.agent_index]
        if "wait" in self.current_ml_action:
            chosen_action = Action.STAY
            return chosen_action
        
        else:
            possible_motion_goals = self.find_motion_goals(state)
            current_motion_goal, chosen_action = self.choose_motion_goal(
                start_pos_and_or, possible_motion_goals)       

        if chosen_action is None:
            self.current_ml_action = "wait(1)"
            # self.time_to_wait = 1
            chosen_action = Action.STAY
        
        return chosen_action



class LLMBot:

    def __init__(self, prompt_path) -> None:
        self.history = []
        self.prompt = ''
        self.history
        self.model = "gpt-3.5-turbo"
        self.temperature = 0.1
        self.top_p = 0.1
        self.max_tokens = 1000
        self.max_history = 10
        self.api_key = "sk-xxx"

        with open(prompt_path) as f:
            self.prompt = f.read()
            self.__add_message("system", self.prompt)

        openai.api_key = self.api_key
        

    def __add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def __call__(self, message: str) -> str:
        self.__add_message("user", message)
        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=self.history[-self.max_history:],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stream=True,
            )
        except Exception as e:
            print(e)
            assert False, f"ERROR: {e}"
        result = []
        for chunk in completion:
            chunk_pieces: dict = json.loads(str(dict(chunk)["choices"][0]["delta"]))
            if "content" in chunk_pieces.keys():
                chunk_content = str(chunk_pieces["content"])
                print(chunk_content, end="", flush=True)
                result.append(chunk_content)
        # print()
        response: str = "".join(result)
        self.__add_message("assistant", response)
        return response



@app.route(f"/<algo>/predict/", methods=["POST"])
def predict(algo):
    if request.method == "POST":
        data_json = json.loads(request.data)
        my_action, state_dict, opp_idx, server_layout_name, timestep = (
            data_json["action"],
            data_json["state"],
            data_json["index"],
            data_json["layout_name"],
            data_json["timestep"],
        )
        timestep = int(timestep)
        print(timestep)
        opp_idx = int(opp_idx)
        layout_name = NAME_TRANSLATION[server_layout_name]
        print(str(state_dict))
        if timestep % 20 == 0:
            print(str(state_dict))
            print("\n\n\n\n\n\n")
            llmres = LLMBots[layout_name](str(state_dict))
            print(llmres)
            LLMAgents[layout_name].current_ml_action = llmres

        # print(state_dict)

        state = process_state(state_dict, layout_name)

        a = LLMAgents[layout_name].action(state)
        # POLICY_P = AGENTS[layout_name][algo]
        # POLICY_P.set_agent_index(npc_index)
        # policy = POLICY_P.direct_action

        # if algo == "bc":
        #     a = BC_ACTION_TRANSLATION[POLICY_P.actions(OvercookedState.from_dict(state_dict))]
        # else:
        #     a = get_prediction(s, policy, layout_name, algo)
        print(f"Player 0 choose action {a}")
    else:
        assert False

    return jsonify({"action": BC_ACTION_TRANSLATION[a]})


@app.route("/beforegame", methods=["POST"])
def beforegame():
    if request.method == "POST":
        f = open('./configs/before_game.yaml')
        config = yaml.load(f, Loader=yaml.FullLoader)
                
    return config

@app.route("/statement", methods=["POST"])
def statement():
    if request.method == "POST":
        f = open('./configs/statement.md', 'r', encoding='utf-8').read()
        html = markdown(f)
    return html


@app.route("/updatemodel", methods=["POST"])
def updatemodel():
    """
    Save game trajectory to file
    """
    if request.method == "POST":
        data_json = json.loads(request.data)
        traj_dict, traj_id, server_layout_name = data_json["traj"], data_json["traj_id"], data_json["layout_name"]
        layout_name = NAME_TRANSLATION[server_layout_name]
        print("saving traj:", traj_id)

        if ARGS.trajs_savepath:
            # Save trajectory (save this to keep reward information)
            filename = f"{traj_id}.json".replace(":", "_")
            save_path = os.path.normpath(ARGS.trajs_savepath)
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.normpath(os.path.join(save_path, filename)).replace("\\", "/"), "w") as f:
                json.dump(traj_dict, f)
                print(f"wrote {filename}")

            # Save transitions minimal (only state/action/done, no reward)
            simultaneous_transitions = convert_traj_to_simultaneous_transitions(traj_dict, layout_name)
            simultaneous_transitions.write_transition(os.path.normpath(os.path.join(save_path, filename)))

        # Finetune model: todo

        REPLAY_TRAJ_IDX = 0
        done = True

    return jsonify({"status": True})


@app.route("/changemodel", methods=["POST"])
def changemodel():
    """
    Change both aagents to allow changing agents on the fly.
    Agents are global variables for now.
    """
    if request.method == "POST":
        data_json = json.loads(request.data)
        print(data_json)
        new_agent_type = int(data_json["agent_type"])
        print("changing model!")
        success = True
        result = {"status": success}

        set_agent_pair(game_settings[new_agent_type]['agents'][0], game_settings[new_agent_type]['agents'][1],
                       game_settings[new_agent_type]['layout'])
        result = {"status": success}

        return jsonify(result)


@app.route("/create_questionnaire_before_game", methods=["POST"])
def create_questionnaire_before_game():
    """
    {
        "name": "Bob",
        "sex": "male",
        "phone": "123456",
        "email":"abc@gmail.com",
        "age" : 20
    }
    Returns:

    """
    os.makedirs(ARGS.questionnaire_savepath, exist_ok=True)
    data_json = json.loads(request.data)
    questionnaire_path = os.path.join(ARGS.questionnaire_savepath, f"{data_json.get('name')}_{data_json.get('phone')}")
    with open(f"{questionnaire_path}.json", "w") as f:
        f.write(json.dumps(data_json))
    return data_json



@app.route("/update_questionnaire_in_game", methods=["POST"])
def create_questionnaire_in_game():
    """
    {
        "name": "Bob",
        "phone": "123456",
        "traj_id":"3_2_2023_9:30:44_human=0",
        "agent_type":"1",
        "questionnaire":{
            "I am playing well.": "I am playing well.",
            "The agent is playing poorly.": "The agent is playing poorly.",
            "The team is playing well.": "The team is playing well.",
    }
    Returns:

    """
    data_json = json.loads(request.data)
    questionnaire_path = os.path.join(ARGS.questionnaire_savepath, f"{data_json.get('name')}_{data_json.get('phone')}")
    with open(f"{questionnaire_path}.json", "r") as f:
        questionnaire = json.load(f)
    if "in_game" not in questionnaire.keys():
        in_game = []
    else:
        in_game = questionnaire["in_game"]
    traj_id = data_json["traj_id"]
    save_path = os.path.normpath(ARGS.trajs_savepath)
    filename = f"{traj_id}.json".replace(":", "_")
    agent_settings_list = list(data_json['agent_settings_list'])
    agent_type_idx = int(data_json["agent_type"])
    try:
        # agent, human = tuple(game_settings[int(data_json["agent_type"])]['agents'])
        agent, human = agent_settings_list[agent_type_idx]['agents']
    except KeyError as e:
        print(e)
        agent, human = None, None

    if human != "human":
        agent, human = human, agent
        human_pos = 0
    else:
        human_pos = 1
    agent_count = 0
    for in_game_item in in_game:
        if in_game_item.get("teammate") == agent:
            agent_count += 1
    in_game.append(
        {
            "traj_path": os.path.normpath(os.path.join(save_path, filename)).replace("\\", "/"),
            "questionnaire": data_json["questionnaire"],
            "teammate": agent,
            "human_pos": human_pos,
            "game_id": f"{agent}_vs_human_{agent_count}",
        }
    )
    questionnaire["in_game"] = in_game
    with open(f"{questionnaire_path}.json", "w") as fw:
        fw.write(json.dumps(questionnaire))
    return questionnaire


@app.route("/update_questionnaire_after_game", methods=["POST"])
def create_questionnaire_after_game():
    """
    {
    "name": "Bob",
    "phone": "123456",
    "questionnaire": {
        "question1": "answer1",
        "question2": "answer2"
        }
    }

    Returns:

    """
    data_json = json.loads(request.data)
    questionnaire_path = os.path.join(ARGS.questionnaire_savepath, f"{data_json.get('name')}_{data_json.get('phone')}")
    with open(f"{questionnaire_path}.json", "r") as f:
        questionnaire = json.load(f)
    after_game = data_json["questionnaire"]
    questionnaire["after_game"] = {"questionnaire": after_game}
    with open(f"{questionnaire_path}.json", "w") as fw:
        fw.write(json.dumps(questionnaire))
    return questionnaire


@app.route("/get_questionnaire", methods=["GET"])
def get_questionnaire():
    """
    "tag": "ingame",

    Returns:

    """
    tag = request.args.get("tag", "")
    if tag == "begin_game":
        return {
            "name": "What is your name/ID?",
            "sex": "Gender?",
            "phone": "For contact maybe a phone number？",
            "email": "For contact maybe a email？",
            "age": "Age",
        }
    elif tag == "after_game":
        return {
            "Which agent cooperates more fluently": "Which agent cooperates more fluently?",
            "Which agent did you prefer playing with": "Which agent did you prefer playing with? ",
            "Which agent did you understand with": "[] Which agent did you understand with? ",
        }
    elif tag == "in_game":
        return {
            "agent_play_well": "The agent is playing well",
            "good_teamwork": "The agent and I have good teamwork/collaboration",
            "agent_contributes_to_success": "The agent is contributing to the success of the team",
            "I_understand_agent": "I understand the agent's intentions",
        }


def init_game_settings_random(algo: str, baselines: list, human_name: str, layouts: list, trial_algo=None,
                              trial_layout='simple', random_start_index=True):
    """
    random_start_index : whether to use randomized starting index in overcooked
    """

    # algo_control = random.choice(baselines)

    ### Choose algo in turn for small-scale test
    global cur_algo_idx
    cur_algo_idx = (cur_algo_idx + 1) % (2*MAX_AGENTS)
    # algo_control = baselines[cur_algo_idx]
    human_idx = cur_algo_idx

    swap_algo_order = random.choice([False, True])  # whether to do ababab or bababa order
    # algo0, algo1 = (algo, algo_control) if not swap_algo_order else (algo_control, algo)

    game_settings = []
    # if trial_algo is not None:
    #     game_settings.append({"agents": [human_name, trial_algo],
    #                           "layout": trial_layout,
    #                           "layout_alias": NAME_TRANSLATION_REVERSE[trial_layout],
    #                           "url": f"http://{LOCAL_IP}:{ARGS.port}/bc/predict/"
    #                           })
    for layout in layouts:
        # do_swap_index = random.choice([0, 1]) if random_start_index else 1s
        # human_algo0_pair = [human_name, algo0] if do_swap_index else [algo0, human_name]
        # human_algo1_pair = [human_name, algo1] if do_swap_index else [algo1, human_name]
        game_settings.extend([{"agents": human_idx,
                               "layout": layout,
                               "layout_alias": NAME_TRANSLATION_REVERSE[layout],
                               "url": f"http://{LOCAL_IP}:{ARGS.port}/0/predict/"
                               }, {
                                  "agents": human_idx,
                                  "layout": layout,
                                  "layout_alias": NAME_TRANSLATION_REVERSE[layout],
                                  "url": f"http://{LOCAL_IP}:{ARGS.port}/0/predict/"
                              }])

    print(human_idx, "\n\n\n\n")
    # print("Algos,  Chosen are: ", algo0, algo1)
    return game_settings


@app.route("/randomize_game_settings", methods=["GET"])
def randomize_game_settings():
    global game_settings
    game_settings = init_game_settings_random(ALGO, ALGO_BASELINES, "human", ALL_LAYOUTS, "bc")
    # print(game_settings)
    return jsonify(game_settings)


@app.route("/")
def root():
    if ARGS.dummy:
        return app.send_static_file("index_dummy.html")
    return app.send_static_file("index_" + "1" + ".html")
    # return app.send_static_file("question.html")


@app.route("/html/<page>")
def return_html(page):
    return app.send_static_file(f"{page}.html")


def load_all_agents(ckpt_paths: list, best_bc_model_paths, load_bc=True):
    """
    We load in all the needed agents when launching server,
    and save them into a dict.
    """
    agents = {
        layout: dict() for layout in ALL_LAYOUTS
    }
    for layout in ALL_LAYOUTS:
        bc_agent, bc_params = get_bc_agent_from_saved(best_bc_model_paths["test"][layout]) if load_bc else None
        agents[layout] = {"bc": bc_agent, "human": None}

        layout_dir = os.path.join(ARGS.ckpts, layout)
        for algo_dir in os.listdir(layout_dir):
            path = os.path.normpath(os.path.join(layout_dir, algo_dir))
            if os.path.isdir(path) and algo_dir in [ALGO] + ALGO_BASELINES:
                agents[layout][algo_dir] = get_agent_from_saved_model(path, ARGS.sim_threads)
                print("loaded ", path)

    return agents


def set_agent_pair(algo0: str, algo1: str, layout_name: str):
    print(algo0, algo1, layout_name)
    set_agent(algo0, 0, layout_name)
    set_agent(algo1, 1, layout_name)


def set_agent(algo, agent_idx: int, layout_name: str):
    """
    Set the two agents to be active (Will be used in the next game)
    """
    print(algo, agent_idx, layout_name)

    global POLICY_P0
    global POLICY_P1
    global ALGO_P0
    global ALGO_P1
    if agent_idx == 0:
        ALGO_P0 = algo
        POLICY_P0 = AGENTS[layout_name][algo]
        if POLICY_P0 is not None:  # if not human keyboard
            # POLICY_P0.set_agent_index(0)
            pass

    if agent_idx == 1:
        ALGO_P1 = algo
        POLICY_P1 = AGENTS[layout_name][algo]
        if POLICY_P1 is not None:  # if not human keyboard
            # POLICY_P1.set_agent_index(1)
            pass

    print(f"set agent {agent_idx} to {algo}")


if __name__ == "__main__":

    game_settings = init_game_settings_random(ALGO, ALGO_BASELINES, "human", ALL_LAYOUTS, "bc")
    print(game_settings)
    # set_agent_pair(AGENT_SETTINGS[ARGS.default][0], AGENT_SETTINGS[ARGS.default][1], LAYOUT_SETTINGS[str(ARGS.default)])
    # print("----------- load agents success! -----------")
    # print(AGENTS)
    # print("---------------------------")

    # TODO: client should pick layout name, instead of server?
    # currently both client/server pick M name, and they must match
    MDPS = {layout_name: OvercookedGridworld.from_layout_name(layout_name=layout_name) for layout_name in ALL_LAYOUTS}
    MLPS = {layout_name: MediumLevelPlanner.from_pickle_or_compute(MDPS[layout_name], NO_COUNTERS_PARAMS, force_compute=False) for layout_name in ALL_LAYOUTS}

    LLMBots = {layout_name: LLMBot("prompt.txt") for layout_name in ALL_LAYOUTS}
    LLMAgents = {layout_name: LLMAgent(MLPS[layout_name], agent_index=1) for layout_name in ALL_LAYOUTS}
    app.run(debug=True, host="0.0.0.0", port=ARGS.port, use_reloader=False)
