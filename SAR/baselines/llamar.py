import re
import base64
import requests
import json, os
import pandas as pd
import tqdm
from pathlib import Path
import sys

# set parent directory to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files
print(os.getcwd())

from misc import Arg

# import environment
from env import SAREnv

# load info about amt of agent before utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=int, default=1)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--action_verbose", action="store_true")
parser.add_argument("--tiny_verbose", action="store_true")
parser.add_argument("--name", type=str, default="default_location")
parser.add_argument("--agents", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# change config file
# write config file
with open("multiagent_config.json", "w+") as f:
    d = {"num_agents": args.agents}
    json.dump(d, f)

# import utils for this baseline - with updated config file
from llamar_utils_multiagent import *
from sar_logging import SARLogger

import warnings
import os


# to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# no warnings
warnings.filterwarnings("ignore")


with open(os.path.expanduser("~") + "/openai_key.json") as json_file:
    key = json.load(json_file)
    api_key = key["my_openai_api_key"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

# environment initialization
env = SAREnv(num_agents=args.agents, scene=args.scene, seed=args.seed, save_frames=True)
env.reset()
timeout = env.task_timeout

# config
config = Arg(temperature=0.7, model="gpt-4")

# logger
logger = SARLogger(env=env, baseline_name=args.name)
print("baseline path (w/ results):", logger.baseline_path)

# some inits
previous_action = [] * args.agents
previous_success = [True, True]  # initialize with True

# there is weird try-except loop wrapper (done in order to prevent json errors)
print("*" * 50)
print("Starting the llamar baseline")
print("*" * 50)

# PLANNER - Initial only!
success = False
while not success:
    try:
        response = get_gpt_response(env, config, action_or_planner="planner")
        outdict = get_action(response)

        if args.verbose:
            print("*" * 10, "planner outdict!", "*" * 10)
            print(outdict)
            print()
        success = True
    except Exception as e:
        print("failure reason (in try-except loop):", e)
        pass

# Start of LoOpP - while not finished or has passed timeout

for step_num in tqdm.trange(timeout):

    update_plan(env, outdict["plan"], env.closed_subtasks)

    # ACTOR
    success = False
    while not success:
        try:
            response = get_gpt_response(env, config, action_or_planner="action")
            outdict = get_action(response)
            preaction, reason, subtask, memory, failure_reason = (
                process_action_llm_output(outdict)
            )
            success = True
            if args.verbose or args.action_verbose:
                print("*" * 10, "Actor outdict!", "*" * 10)
                print(outdict)
                print()
        except Exception as e:
            print("failure reason (in try-except loop):", e)
            pass

    action = action_mapping(env, preaction)

    logger.log_agent_mem(env.step_num, action, reason, subtask, memory)
    # MISSING: conditional (we're using this now)
    env.update_subtask(subtask, 0)
    env.update_memory(memory, 0)
    d1, successes = env.step(action)
    previous_action = action
    previous_success = successes

    # VERIFIER
    success = False
    while not success:
        try:
            response = get_gpt_response(env, config, action_or_planner="verifier")
            outdict = get_action(response)
            if args.verbose:
                print("*" * 10, "verifier outdict!", "*" * 10)
                print(outdict)
                print()
            success = True
        except Exception as e:
            print("failure reason (in try-except loop):", e)
            pass

    # v0 - update completed subtasks list - no for now
    # env.closed_subtasks = set_addition(env.closed_subtasks, outdict["completed subtasks"])
    env.closed_subtasks = outdict["completed subtasks"]
    if len(env.closed_subtasks) == 0:
        env.closed_subtasks = None
    env.input_dict["Robots' completed subtasks"] = env.closed_subtasks
    env.get_planner_llm_input()

    # PLANNER
    success = False
    while not success:
        try:
            response = get_gpt_response(env, config, action_or_planner="planner")
            outdict = get_action(response)
            success = True
        except Exception as e:
            print("failure reason (in try-except loop):", e)
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

if args.tiny_verbose:
    # get statistics and finish step
    coverage = env.checker.get_coverage()
    transport_rate = env.checker.get_transport_rate()
    finished = env.checker.check_success()

    print("_" * 50)
    print(f"Step {step_num}")
    print(f"Completed Subtasks: ")
    print("\n".join(env.checker.subtasks_completed))
    print("Transport rate:", transport_rate)
    print("Coverage:", coverage)
    print("Finished:", finished)


# STOP - stop the controller / remove window
env.stop()
