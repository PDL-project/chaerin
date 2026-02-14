import re
import base64
import requests
import json, os
import pandas as pd
import tqdm
from pathlib import Path
import sys

# --- DESCRIPTION ---
# llava.py is the main script that runs the llava VLM baseline in AI2Thor on the floorplans and tasks specified in the command line using the utils and definitions defined in llava_utils.py

# set parent directory to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files
print(os.getcwd())

# import environment
from AI2Thor.env_new import AI2ThorEnv

# import utils for this baseline
from llava_utils import *
from AI2Thor.baselines.utils import Logger, AutoConfig

import warnings
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=int, default=0)
parser.add_argument("--floorplan", type=int, default=0)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--name", type=str, default="idefics2")
args = parser.parse_args()

# to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# no warnings
warnings.filterwarnings("ignore")


with open(os.path.expanduser("~") + "/openai_key.json") as json_file:
    key = json.load(json_file)
    api_key = key["my_openai_api_key"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

# autoconfig
auto = AutoConfig()
auto.set_task(args.task)
auto.set_floorplan(args.floorplan)
timeout = auto.get_task_timeout()

# environment initialization
config = auto.config()

env = AI2ThorEnv(config)
d = env.reset(task=auto.task_string())
# logger
logger = Logger(env=env, baseline_name=args.name)
print("baseline path (w/ results):", logger.baseline_path)

# some inits
previous_action = [] * config.num_agents
previous_success = [True, True]  # initialize with True

print("*" * 50)
print("Starting the llamar baseline")
print("*" * 50)

# PLANNER - Initial only!
success = False
while not success:
    try:
        response = get_llava_response(env, config, action_or_planner="planner")
        outdict = get_action(response)
        if args.verbose:
            print("Planner Output:\n", outdict)
        success = True
    except Exception as e:
        print(e)
        pass

# Execute VLM planner on specified task in specified floorplan:

for step_num in tqdm.trange(timeout):

    update_plan(env, outdict["plan"], env.closed_subtasks)

    # ACTOR
    success = False
    while not success:
        try:
            response = get_llava_response(env, config, action_or_planner="action")
            outdict = get_action(response)
            success = True
        except:
            pass

    preaction, reason, subtask, memory, failure_reason = process_action_llm_output(
        outdict
    )
    action = print_stuff(env, preaction, reason, subtask, memory, failure_reason)

    logger.log_agent_mem(env.step_num, action, reason, subtask, memory)
    # manual temporary logging
    if config.use_shared_subtask:
        env.update_subtask(subtask, 0)
    if config.use_shared_memory:
        env.update_memory(memory, 0)
    d1, successes = env.step(action)
    previous_action = action
    previous_success = successes
    if args.verbose:
        print_relevant_info(env, config, env.input_dict)

    # VERIFIER
    success = False
    while not success:
        try:
            response = get_llava_response(env, config, action_or_planner="verifier")
            outdict = get_action(response)
            if args.verbose:
                print("Verifier Output:\n", outdict)
            success = True
        except:
            pass

    # v0 - update completed subtasks list - no for now
    env.closed_subtasks = outdict["completed subtasks"]
    if len(env.closed_subtasks) == 0:
        env.closed_subtasks = None
    env.input_dict["Robots' completed subtasks"] = env.closed_subtasks
    env.get_planner_llm_input()

    # PLANNER
    success = False
    while not success:
        try:
            response = get_llava_response(env, config, action_or_planner="planner")
            outdict = get_action(response)
            success = True
        except:
            pass

    # get statistics and finish step
    coverage = env.checker.get_coverage()
    transport_rate = env.checker.get_transport_rate()
    finished = env.checker.check_success()

    # log current 'step'
    logger.log_step(
        step=step_num,
        preaction=preaction,
        action=action,
        success=previous_success,
        coverage=coverage,
        transport_rate=transport_rate,
        finished=finished,
    )

    if args.verbose:
        print("_" * 50)
        print(f"Step {step_num}")
        print(f"Completed Subtasks: ")
        print("\n".join(env.checker.subtasks_completed))

    # if the model outputs "Done" for both agents, break
    if all(status == "Done" for status in action):
        break

# STOP - stop the controller / remove window
env.controller.stop()
