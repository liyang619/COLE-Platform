import $ from "jquery"
import _ from "lodash"

import OvercookedSinglePlayerTask from "./js/overcooked-single";
import getOvercookedPolicy from "./js/load_tf_model.js";

import * as Overcooked from "overcooked"
let OvercookedMDP = Overcooked.OvercookedMDP;
let Direction = OvercookedMDP.Direction;
let Action = OvercookedMDP.Action;
let [NORTH, SOUTH, EAST, WEST] = Direction.CARDINAL;
let [STAY, INTERACT] = [Direction.STAY, Action.INTERACT];

// Parameters
let PARAMS = {
    MAIN_TRIAL_TIME: 15, //seconds
    TIMESTEP_LENGTH: 150, //milliseconds
    DELIVERY_POINTS: 20,
    PLAYER_INDEX: 1,  // Either 0 or 1
    MODEL_TYPE: 'ppo_bc'  // Either ppo_bc, ppo_sp, or pbt
};

/***********************************
      Main trial order
************************************/


let layouts = {
    "cramped_room":[
        "XXPXX",
        "O  2O",
        "X1  X",
        "XDXSX" 
    ],
    "asymmetric_advantages":[
        "XXXXXXXXX",
        "O XSXOX S",
        "X   P 1 X",
        "X2  P   X",
        "XXXDXDXXX"
    ],
    "coordination_ring":[
        "XXXPX",
        "X 1 P",
        "D2X X",
        "O   X",
        "XOSXX"
    ],
    "forced_coordination":[
        "XXXPX",
        "O X1P",
        "O2X X",
        "D X X",
        "XXXSX"
    ],
    "counter_circuit": [
        "XXXPPXXX",
        "X      X",
        "D XXXX S",
        "X2    1X",
        "XXXOOXXX"
    ],
    "fork": [
        "XXXXXXXXXXXXXXX",
        "XXXXX  X    XPX",
        "O1    XXXXX  2S",
        "XXXXX  X    XPX",
        "XXXXXXXXXXXXXXX"
    ]
};

let agent_settings = {
    "0" : ["human", "ai"],
    "1" : ["human", "ai"],
    "2" : ["ai", "ai"]
}
// should be in line with app.py

let game;
let agentTypeSelector;

document.addEventListener("DOMContentLoaded", function() {
    // 前端逻辑：每一次load，先调接口随机给一个agent_type，完成一关之后（到时间后），跳转到问卷页面，填写问卷后继续游戏，最后完成所有游戏之后填写整个游戏反馈的页面
    agentTypeSelector = document.getElementById("agent_type");
    console.log('1111111')
    agentTypeSelector.addEventListener('change', event => {
        changeModel($("#agent_type").val());
    });
});


function startGame(endOfGameCallback) {
    console.log('222222')
    let AGENT_INDEX = 1 - PARAMS.PLAYER_INDEX;
    /***********************************
          Set up websockets server
    ***********************************/
    // let HOST = "https://lit-mesa-15330.herokuapp.com/".replace(/^http/, "ws");
    // let gameserverio = new GameServerIO({HOST});

    // let players = [$("#playerZero").val(), $("#playerOne").val()];
    let players = agent_settings[$("#agent_type").val()]
    console.log(players)
    let algo = $("#algo").val();
    let deterministic = $("#deterministic").is(':checked');
    let saveTrajectory = $("#saveTrajectories").is(':checked');
    if (players[0] == 'human' && players[1] == 'human') {
        $("#overcooked").html("<h3>Sorry, we can't support humans as both players.  Please make a different dropdown selection and hit Enter!</h3>"); 
        endOfGameCallback();
        return;
    }
    let player_index = null; 
    let npc_policies = [0, 1]; 
    if (players[0] == 'human') {
        player_index = 0;
        npc_policies = [1];
    }
    if (players[1] == 'human') {
        player_index = 1;
        npc_policies = [0]; 
    }
    
    let layout_name = $("#layout").val();
    let game_time = $("#gameTime").val();
    if (game_time > 1800) {
        $("#overcooked").html("<h3>Sorry, please choose a shorter amount of time for the game!</h3>"); 
        endOfGameCallback();
        return;
    }

    let layout = layouts[layout_name];
    console.log(layout)
    console.log(layout_name)


    $("#overcooked").empty();
    let mdp_params = {
        "layout_name": layout_name, 
        "num_items_for_soup": 3, 
        "rew_shaping_params": null, 
    }
    game = new OvercookedSinglePlayerTask({
        container_id: "overcooked",
        player_index: player_index,
        start_grid: layout,
        layout_name: layout_name,
        npc_policies: npc_policies,
        algo: algo,
        mdp_params: mdp_params,
        task_params: PARAMS,
        save_trajectory: saveTrajectory,
        TIMESTEP : PARAMS.TIMESTEP_LENGTH,
        MAX_TIME : game_time, //seconds
        init_orders: null,
        completion_callback: () => {
            console.log("Time up");
            // give a questionaire
            endOfGameCallback();
        },
        DELIVERY_REWARD: PARAMS.DELIVERY_POINTS
    });
    game.init();
}

function endGame() {
    game.close();
}

// Handler to be added to $(document) on keydown when a game is not in
// progress, that will then start the game when Enter is pressed.
function startGameOnEnter(e) {
    // Do nothing for keys other than Enter
    if (e.which !== 13) {
	return;
    }

    disableEnter();
    // Reenable enter handler when the game ends
    startGame(enableEnter);
}

function enableEnter() {
    $(document).keydown(startGameOnEnter);
    $("#control").html("<p>Press enter to begin!</p><p>(make sure you've deselected all input elements first!)</p>");
}

function disableEnter() {
    $(document).off("keydown");
    $("#control").html('');
    // $("#control").html('<button id="reset" type="button" class="btn btn-primary">Reset</button>');
    // $("#reset").click(endGame);
}


function changeModel(agent_type) {
    console.log("Changed Model to: " + agent_type);
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/changemodel", false); // false for synchronous
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(JSON.stringify({
        agent_type : agent_type
    }));
    var response = JSON.parse(xhr.response);
    console.log("response:")
    console.log(response);
}

function getRandomAgent() {

}
function isAgentPlayedTest(agent) {

}
function saveDataToSession() {
    
}

$(document).ready(() => {
    enableEnter();
});



