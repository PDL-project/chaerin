import base64
import requests
import json, os
import pandas as pd
from pathlib import Path
import sys

# set parent directory to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files
print(os.getcwd())

from env import SAREnv
from base_env import SARBaseEnv as baseenv
from object_actions import get_closest_feasible_action
from misc import *

with open(os.path.expanduser("~") + "/openai_key.json") as json_file:
    key = json.load(json_file)
    api_key = key["my_openai_api_key"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

# read config file
with open("multiagent_config.json", "r") as f:
    d = json.load(f)
    NUM_AGENTS = d["num_agents"]

# AGENT_NAMES global variable from env_new contains all the agent names (6 of them)
# subsample to num agents (so len(.) gives accurate amt)
AGENT_NAMES_ALL = SAREnv.AGENT_NAMES
AGENT_NAMES = AGENT_NAMES_ALL[:NUM_AGENTS]

# useful variables
FIRE_TYPES = SAREnv.FIRE_TYPES
EXTINGUISH_TYPES = SAREnv.EXTINGUISH_TYPES
AMT_FIRE_TYPES = len(FIRE_TYPES)
INVENTORY_CAPACITY = SAREnv.INVENTORY_CAPACITY
INVENTORY_TYPES = SAREnv.INVENTORY_TYPES
MIN_REQUIRED_AGENTS = SAREnv.MIN_REQUIRED_AGENTS
ALL_INTENSITIES = SAREnv.ALL_INTENSITIES
CRITICAL_INTENSITY = SAREnv.CRITICAL_INTENSITY
L_TO_M = SAREnv.L_TO_M
M_TO_H = SAREnv.M_TO_H
CARDINAL_DIRECTIONS = SAREnv.CARDINAL_DIRECTIONS
GRID_WIDTH = SAREnv.GRID_WIDTH
GRID_HEIGHT = SAREnv.GRID_HEIGHT

# NOTE: removed information about when to use deposit (just avoid altogether)
_removed = """
Therefore, since time is of the essence to extinguish fires, it might be benefitial to have some robots collecting resources for others and store it in deposits.
This is in order to avoid having the ones fighting the fires have to waste steps collecting the resources 1-unit at a time from the reservoirs.
However, if an agent goes to the deposit, drops their supplies, and then immediately collects them, this is a waste of time. The use of deposits only makes sense if some agents are 'collectors' and other are 'firefighters'. So make sure you use deposits wisely.
"""

# v1 - updated environment string to also give the negatives of using deposits (intended, more neutral)
# v2 - add rules about how the env processes person drops
# v3 - removed info about why to use/not to use, made it clear to steer away of deposits to store resources
# v4 - add info that fire has different regions and they're all important
# v5 - some regions might be on fire and some not (not all equally important)

ENV_STR = f"""The environment consists of fires and lost persons, along with reservoirs, deposits, and robots (you). All in a grid with width {GRID_WIDTH} and height {GRID_HEIGHT}.

Initially, the robots can see all the fires, but does not know the location of any of the lost people - robots must explore.
The fires can be of {AMT_FIRE_TYPES} different types: {join_conjunction(FIRE_TYPES, 'or')}, each requiring a different resource to extinguish - {join_conjunction(EXTINGUISH_TYPES, 'and')} respectively. Make sure you use the proper resource to do so.
A fire consists of a group of 'flammable' objects with intensities of {join_conjunction(ALL_INTENSITIES, 'or')}. It is divided into different regions geographically, so all regions that aren't extinguished (intensity {ALL_INTENSITIES[0]}), must be properly addressed before fire can be extinguished. The first few regions (1,2,etc) are the sources of the fire, and must be addressed first.
At each step, if a flammable object has an intensity of {ALL_INTENSITIES[1]} or higher, it'll increase in intensity if not extinguished. They spread quickly, so it's important for almost all robots to work collectively to stop the fire.
In {L_TO_M} steps, the flammable object will go from {ALL_INTENSITIES[1]} to {ALL_INTENSITIES[2]}. In {M_TO_H} steps, the flammable object will go from {ALL_INTENSITIES[2]} to {ALL_INTENSITIES[3]}.
Once a flammable object reaches an intensity of {CRITICAL_INTENSITY} and not before, it spreads to its immediate neighboors (neighboors with intensity {ALL_INTENSITIES[0]} start with intensity of {ALL_INTENSITIES[1]}).  In order to extinguish a fire, a robot can use the appropriate extinguish resource at that location.
Then, all the flammable objects in or immediately around this location will lower in intensity by one notch (e.g. from {ALL_INTENSITIES[2]} to {ALL_INTENSITIES[1]}, or {ALL_INTENSITIES[3]} to {ALL_INTENSITIES[2]}).

The reservoirs can be of type {join_conjunction(EXTINGUISH_TYPES, 'or')}, resources can only be collected at a rate of 1-unit per step.
Thus, to get more resources, you have to collect the resources multiple times.

The deposits can hold any amount and type of resources: {join_conjunction(INVENTORY_TYPES, 'and')}.
The robots can store their entire inventory into the deposit in order to save it for other robots to use.
When a robot gets a certain resource type (if any is available) from a deposit, the space left in their inventory is filled with that resource type.
Deposits create an unnecessary middle-step when using them to store resources, so do not waste time in this way.

If any robot enters the area of visibility of a lost person, that person is found and all robots can now see it.
Once a person is found, at least {MIN_REQUIRED_AGENTS} robots are required to carry it (could be more depending on person). Otherwise, the person cannot be moved.
A carried person should be dropped into a deposit (any suffices).
To drop a carried person, all agents should have navigated to deposit, and they must ALL perform the DropOff action.

The robots have an inventory capacity of {INVENTORY_CAPACITY} with slots for {join_conjunction(INVENTORY_TYPES, 'and')}."""

OBS_STR = f"""You will get a description of the task robots are supposed to do. You will get an textual description of the environment from the perspective of {join_conjunction(AGENT_NAMES, 'and')} as the observation input. You will also get a list of objects each robot is able to see in the environment. Here the objects will have a distinct name which will also include which type of object it is.
So, along with the observation inputs you will get the following information:
"""


# planner
# REMOVED: "do not perform additional steps..."
# REMOVED: "since there are _ agents, ... don't bump into each other"
# v1 - Avoid having too fine-grained tasks (especially if there might the multiple of them)
# v2 - divide subtasks into extinguishing regions and extinguishing overall fire

PLANNER_OBS_STR = ",\n".join(
    [
        f"{name}'s observation: local observation (from up, down, left, right, center), global observation, and a list of objects {name} is observing"
        for name in AGENT_NAMES
    ]
)

PLANNER_PROMPT = f"""You are an excellent planner who is tasked with helping {len(AGENT_NAMES)} embodied robots named {join_conjunction(AGENT_NAMES, 'and')} to carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

{ENV_STR}

{OBS_STR}
### INPUT FORMAT ###
{{Task: description of the task the robots are supposed to do,
{PLANNER_OBS_STR},
Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' combined memory: description of robots' combined memory}}

Reason over the robots' task, image inputs, observations, open subtasks, completed subtasks and memory, and then output the following:
* Reason: The reason for why new subtasks need to be added.
* Subtasks: A list of open subtasks the robots are supposed to take to complete the task. Remember, as you get new information about the environment, you can modify this list. You can keep the same plan if you think it is still valid. Do not include the subtasks that have already been completed.
The "Plan" should be in a list format where the actions are listed sequentially.
For example:
     ["extinguish ChicagoFire_Region_1 using water", "extinguish ChicagoFire_Region_2 using water", "extinguish all of ChicagoFire using water", "collect sufficient water form reservoir"]
     ["locate the lost person", "carry the lost person", "navigate to deposit with lost person", "drop lost person in deposit"]

Your output should be in the form of a python dictionary as shown below.
Example output: {{
"reason": "since the subtask list is empty, the robots need to extinguish the fire, and find & drop the lost person. Thus, for the fire we have to get water from the reservoir and extinguish the chicago fire, using the deposit if needed to store resources. For the person, we have to locate them by exploring, carry them with enough agents, go to a deposit, and drop the person in it.",
"plan": ["extinguish ChicagoFire_Region_1 using water", "extinguish ChicagoFire_Region_2 using water", "extinguish all of ChicagoFire using water", "locate the lost person", "carry the lost person", "navigate to deposit with lost person", "drop lost person in deposit"]
}}

Ensure that the subtasks are not generic statements like "explore the environment" or "do the task". They should be specific to the task at hand.
Do not assign subtasks to any particular robot. Try not to modify the subtasks that already exist in the open subtasks list. Rather add new subtasks to the list.

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
Let's work this out in a step by step way to be sure we have the right answer.
"""


# verifier
# moves subtasks from open to completed
# v1 - don't have ambiguous / hyper-specific tasks ('collect enough water' or 'use reservoir'). Add only those that are clear

VERIFIER_OBS_STR = ",\n".join(
    [
        f"{name}'s observation: list of objects the {name} is observing,\n{name}'s state: description of {name}'s state,\n{name}'s previous action: the action {name} took in the previous step,"
        for name in AGENT_NAMES
    ]
)

# v1 - disclaimer at end to not add completed subtask before it has been (try to avoid catastrophic forgetting)
VERIFIER_PROMPT = f"""You are an excellent planner who is tasked with helping {len(AGENT_NAMES)} embodied robots named {join_conjunction(AGENT_NAMES, 'and')} to carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

{ENV_STR}

{OBS_STR}
### INPUT FORMAT ###
{{Task: description of the task the robots are supposed to do,
{VERIFIER_OBS_STR}
Robots' open subtasks: list of open subtasks the robots in the previous step. If no plan has been already created, this will be None,
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None,
Robots' combined memory: description of robots' combined memory
}}

You will receive the following information:
* Reason: The reason for why you think a particular subtask should be moved from the open subtasks list to the completed subtasks list.
* Completed Subtasks: The list of subtasks that have been completed by the robots. Note that you can add subtasks to this list only if they have been successfully completed and were in the open subtask list. If no subtasks have been completed at the current step, return an empty list.

The "Completed Subtasks" should be in a list format where the completed subtasks are listed. For example: ["collect water from deposit", "collect sufficient sand from reservoir"]
Your output should be in the form of a python dictionary as shown below.

Example output with two agents (do it for {len(AGENT_NAMES)} agents):
{{"reason": "{AGENT_NAMES[0]} used water on the ChicagoFire in the previous step and was successful, and {AGENT_NAMES[1]} explored and was successful. Since the ChicagoFire is still not extinguished completely, {AGENT_NAMES[0]} has still not completed the subtask of extinguishing ChicagoFire using water. Since the lost person is now visible {AGENT_NAMES[1]} has completed the subtask of finding the lost person.",
"completed subtasks": ["locate lost person"]
}}

When you output the completed subtasks, make sure to not forget to include the previous ones in addition to the new ones.
Also, make sure to never add a subtask to completed subtasks before it has successfully been completed.
Let's work this out in a step by step way to be sure we have the right answer.

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
"""

# v1 - change failure reasons to include causal order and relevant information
FAILURE_REASON = """
If any robot's previous action failed, use the previous history, your current knowledge of the room (i.e. what things are where), and your understanding of causality to think and rationalize about why the previous action failed. Output the reason for failure and how to fix this in the next timestep. If the previous action was successful, output "None".
Common failure reasons to lookout for include:
Trying to collect a resource before navigating to reservoir first,
trying to use a resource before navigating to fire location first,
not being close enough to interact with object.
"""

# NEW example
# changed to non-related, yet specific example

action_wrapper = lambda name, action: f'"{name}\'s action" : "{action}"'

# previous actions (that failed) -> ["DropOff(LostPersonJeremy, TheDeposit)", "Idle", "UseSupply(ChicagoFire, Sand)"]
action_agents_1 = ["NavigateTo(TheDeposit)", "Idle", "UseSupply(ChicagoFire, Water)"]
ACTION_1 = ",\n".join(
    action_wrapper(AGENT_NAMES_ALL[i], action_agents_1[i]) for i in range(3)
)

# ----- example 1 (failure) ------
# "failure reason" - 1
FAILURE_REASON_EX_1 = "".join(
    [
        f"{AGENT_NAMES_ALL[0]} and {AGENT_NAMES_ALL[1]} failed to drop off LostPersonJeremy in TheDeposit because {AGENT_NAMES_ALL[1]} had not navigated to the deposit yet, and thus wasn't close enough to interact with it; they both have to be close enough to deposit.",
        f"{AGENT_NAMES_ALL[2]} failed to use the sand supply on the ChicagoFire because the fire is non-chemical, so it requires water.",
    ]
)

# "memory" - 1
MEMORY_EX_1 = " ".join(
    [
        f"{AGENT_NAMES_ALL[0]} finished trying to DropOff LostPersonJeremy at the TheDeposit when {AGENT_NAMES_ALL[0]} was at co-ordinates (4,4).",
        f"{AGENT_NAMES_ALL[1]} finished being idle when {AGENT_NAMES_ALL[1]} was at co-ordinates (14, 6).",
        f"{AGENT_NAMES_ALL[2]} finished using sand supply on the ChicagoFire when {AGENT_NAMES_ALL[2]} was at co-ordinates (7, 24).",
    ]
)

# "reason" - 1
REASON_EX_1 = " ".join(
    [
        f"{AGENT_NAMES_ALL[0]} can wait for {AGENT_NAMES_ALL[1]} to finish navigating to TheDeposit.",
        f"{AGENT_NAMES_ALL[1]} can navigate to TheDeposit in order to be close enough to DropOff LostPersonJeremy.",
        f"{AGENT_NAMES_ALL[2]} can go to use their water supply instead on the ChicagoFire.",
    ]
)

# "subtask" - 1
SUBTASK_EX_1 = " ".join(
    [
        f"{AGENT_NAMES_ALL[0]} is currently waiting for {AGENT_NAMES_ALL[1]} to finish navigating,",
        f"{AGENT_NAMES_ALL[1]} is currently navigating to TheDeposit,",
        f"{AGENT_NAMES_ALL[2]} is currently navigating to the EasternFire,",
    ]
)


# -- construct failure example from this ---
FAILURE_EXAMPLE = f"""
Example:
{{
"failure reason": "{FAILURE_REASON_EX_1}",
"memory": "{MEMORY_EX_1}",
"reason": "{REASON_EX_1}",
"subtask": "{SUBTASK_EX_1}",
{ACTION_1}
}}
"""

ACTION_OBS_STR = ", ".join(
    [
        f"{name}'s observation: list of objects the {name} is observing,\n{name}'s state: description of {name}'s state,\n{name}'s previous action: description of what {name} did in the previous time step and whether it was successful,\n{name}'s previous failures: if {name}'s few previous actions failed, description of what failed,"
        for name in AGENT_NAMES
    ]
)

# action planner
# v1 - added that it must navigate to deposit strictly before dropping the agents
# v2 - added info about constraints in dropping a person
# v3 - added info about constraints on how many supplies dropped (must do multiple to clear inventory)
# v4 - change order, emphasize navigate before interact rule (does it think it includes use supply?)
# v5 - be specific for region to navigate to
# v6 - tell it that you can't direct use supply, it's only where you are

# details for actor
DETAILS_STR = f"""
Important details described below:
    * Even if the robot can see an object, it might not be able to interact with them if they are too far away. Hence you will need to make the robot navigates to the objects they want to interact with.
    * When navigating to fire, please specify which specific region of the fire you wish to target.
    * Additionally, when you use the supply, it will be dropped wherever you are, NOT in the region you said. So, make sure to navigate to wherever you wish to use a supply.
    * When a fire has an average intensity of {ALL_INTENSITIES[0]}, it means all the flammable objects have been extinguished completely.
    * When the person is not initially visible, you must Explore.
    * When a person is carried, no other action other than navigation and "DropOff" can be made by any of the robots carrying it.
    * When a robot carries a person, all their other resources are dropped and the person takes the entire inventory space.
    * When a group robot wants to drop a person, they must all navigate to the deposit strictly before dropping them. Then, they should ALL perform the DropOff action.
    * When a robot is successful in carrying a person, that just means that specific robot is carrying it, but the person might still not be moveable if an insufficient amount of robots is carrying it.
    * A fire is divided into different regions, and the fire itself (which is just the center). However, there might be flammable objects from the fire that aren't immediately neighbooring this location, so the robot might have to move in different directions to reach them.
    * The resources in the inventory can only be used on fires one unit at a time, so do multiple UseSupply until you clear our your inventory.
    * When a robot is doing a "Move(<direction>)" action, if there is an obstacle, they will not be able to move in that direction.
"""


# @here
# v1 - added more specific output than 'action' to avoid dict error
# v2 - added "Done" action
# v3 - added that when using multiple agents, it must address *different* things (like different fires) + avoid agents being idle

ACTION_PROMPT = f"""
You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {join_conjunction(AGENT_NAMES, 'and')} carry out a task. All {len(AGENT_NAMES)} robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

{ENV_STR}

They can perform the following actions:
["navigate to object <object_name>", "move <direction>", "explore", "carry <person_name>", "dropoff <person_name> at <deposit_name>", "store supply <deposit_name>", "use supply <resource_name> on <fire_name>", "get supply <resource_name> from <deposit_name>", "get supply <reservoir_name>", "clear inventory", "Done"]

Here <direction> is one of {str(CARDINAL_DIRECTIONS)}.
Here <resource_name> is one of {str(EXTINGUISH_TYPES)}.
The other names (<object_name>, <person_name>, <deposit_name>, <reservoir_name>) are based on the observations.
When finished with all the subtasks, output "Done" for all agents.

You need to suggest the action that each robot should take at the current time step.

{OBS_STR}
### INPUT FORMAT ###
{{Task: description of the task the robots are supposed to do,
{ACTION_OBS_STR}
Robots' open subtasks: list of subtasks  supposed to carry out to finish the task. If no plan has been already created, this will be None.
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' subtask: description of the subtasks the robots were trying to complete in the previous step,
Robots' combined memory: description of robot's combined memory}}

First of all you are supposed to reason over the image inputs, the robots' observations, previous actions, previous failures, previous memory, subtasks and the available actions the robots can perform, and think step by step and then output the following things:
* Failure reason: {FAILURE_REASON}
* Memory: Whatever important information about the scene you think you should remember for the future as a memory. Remember that this memory will be used in future steps to carry out the task. So, you should not include information that is not relevant to the task. You can also include information that is already present in its memory if you think it might be useful in the future.
* Reason: The reasoning for what each robot is supposed to do next
* Subtask: The subtask each robot should currently try to solve, choose this from the list of open subtasks.
* Actions for join_conjunction(AGENT_NAMES, 'and'): The actions the robots are supposed to take just in the next step such that they make progress towards completing the task. Make sure that this suggested actions make these robots more efficient in completing the task as compared only one agent solving the task, avoid having agents being idle. Notably, make sure that different agents address different problems (e.g. different fires)
Your output should just be in the form of a python dictionary as shown below.

Example of output for 3 robots (do these for {len(AGENT_NAMES)} robots):
{FAILURE_EXAMPLE}
Note that the output should just be a dictionary similar to the example output.

{DETAILS_STR}
"""

"""
print("*"*30+"\nPLANNER PROMPT:\n"+"*"*30)
print(PLANNER_PROMPT)
print("*"*30+"\nVERIFIER PROMPT:\n"+"*"*30)
print(VERIFIER_PROMPT)
print("*"*30+"\nACTION PROMPT:\n"+"*"*30)
print(ACTION_PROMPT)
"""


def process_action_llm_output(outdict):
    action = []
    for i in range(len(AGENT_NAMES)):
        action.append(outdict[f"{AGENT_NAMES[i]}'s action"])
    reason = outdict["reason"]
    subtask = outdict["subtask"]
    memory = outdict["memory"]
    failure_reason = outdict["failure reason"]
    return action, reason, subtask, memory, failure_reason


def action_mapping(env, action):
    action = action_checker(env, action)
    return action


def encode_image(image_path: str):
    # if not os.path.exists()
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def prepare_prompt(env, module_name: str, addendum: str):
    """module_name: str
    choose from planner, verifier, action
    """
    # Choose the appropriate prompt based on what module is being called
    # user_prompt = baseenv.convert_dict_to_string(env.input_dict)
    if module_name == "action":
        system_prompt = ACTION_PROMPT
        user_prompt = baseenv.convert_dict_to_string(env.get_action_llm_input())
    elif module_name == "planner":
        system_prompt = PLANNER_PROMPT
        user_prompt = baseenv.convert_dict_to_string(env.get_planner_llm_input())
    elif module_name == "verifier":
        system_prompt = VERIFIER_PROMPT
        user_prompt = baseenv.convert_dict_to_string(env.get_verifier_llm_input())
    user_prompt += addendum
    return system_prompt, user_prompt


# NOTE: No images here
def prepare_payload(env, config, module_name: str, addendum: str = ""):
    """# payload consists of
    * the system prompt (which is constant)
    * the user prompt (which changes based on the state)
    This is then sent to the openai api to get the response (action or plan or verification of the plan)
    """
    system_prompt, user_prompt = prepare_prompt(env, module_name, addendum)
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "max_tokens": 1000,
        "temperature": config.temperature,
    }
    return payload


def get_action(response):
    response_dict = response.json()
    # convert the string to a dict
    # json_acceptable_string = response_dict["choices"][0]["message"]["content"].replace("'", "\"").replace("\n", "").replace("json", "").replace("`", "")
    output = response_dict["choices"][0]["message"]["content"]
    json_match = re.search(r"```json(.*?)```", output, re.DOTALL)
    python_match = re.search(r"```python(.*?)```", output, re.DOTALL)
    tilde_match = re.search(r"```(.*?)```", output, re.DOTALL)
    if json_match:
        # print("Got JSON TYPE OUTPUT")
        json_data = json_match.group(1)
    elif python_match:
        # print("Got JSON TYPE OUTPUT")
        json_data = python_match.group(1)
    elif tilde_match:
        # print("Got JSON TYPE OUTPUT")
        json_data = tilde_match.group(1)
    else:
        # print("Got NORMAL TYPE OUTPUT")
        json_data = output
    # print(json_data)
    out_dict = json.loads(json_data)
    return out_dict


def get_gpt_response(env, config, action_or_planner: str, addendum: str = ""):
    payload = prepare_payload(env, config, action_or_planner, addendum)
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response


def action_checker(env, actions):
    """
    Get closest valid action

    The action output from the model is in natural language.
    This function will find the env feasible action which has the closest embedding
    to the natural language action output from the model.
    Eg: "pick up the apple" -> "PickupObject(Apple_1)"
    """
    checked_actions = []
    for act in actions:
        act = get_closest_feasible_action(act, env.object_dict)
        checked_actions.append(act)
    return checked_actions


# v0 - change it so that it appends instead
# for verifier
def set_addition(l1, l2):
    if l1 is None:
        l1 = []
    if l2 is None:
        l2 = []
    return list(set(l1 + l2))


def update_plan(env, open_subtasks, completed_subtasks):
    env.open_subtasks = open_subtasks
    env.closed_subtasks = completed_subtasks
    env.input_dict["Robots' open subtasks"] = env.open_subtasks
    env.input_dict["Robots' completed subtasks"] = env.closed_subtasks
