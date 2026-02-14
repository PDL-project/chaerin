import json, os
import pandas as pd
import tqdm
from pathlib import Path
import sys

# set directory to top folder level to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files

# import environment
from AI2Thor.env_new import AI2ThorEnv

# import utils for this baseline
from AI2Thor.baselines.utils import Logger, AutoConfig
from AI2Thor.baselines.smartllm.smartllm_utils import *

import warnings
import time
import os

import argparse  

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=int, default=0)
parser.add_argument("--floorplan", type=int, default=0)
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument(
    "--partial_obs", action="store_true", default=False, help="Partial Observations"
)
parser.add_argument("--config_file", type=str, default="config.json")
args = parser.parse_args()

# to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# no warnings
warnings.filterwarnings("ignore")


# autoconfig
auto = AutoConfig(config_file=args.config_file)
auto.set_task(args.task)
auto.set_floorplan(args.floorplan)
timeout = auto.get_task_timeout()

# environment initialization
config = auto.config()

a = time.time()

env = AI2ThorEnv(config)
d = env.reset(task=auto.task_string())
partial_obs = args.partial_obs
partial_obs_str = "partial_obs" if partial_obs else "full_obs"
# logger
logger = Logger(env=env, baseline_name=f"SmartLLM_{partial_obs_str}")

b = time.time()

# some inits
previous_action = [] * config.num_agents
previous_success = [True, True]  # initialize with True
df = pd.DataFrame(
    columns=["Step", "Action", "Message", "Random Flag", "Pre Random Action"]
)

agent = SmartLLMAgent(config, env, partial_obs=True)
# p = agent.get_decompose_module_prompt()


decomposed_plan, allocated_plan, code_plan = agent.generate_solution()

dir_path = logger.baseline_path / logger.env.action_dir_path
dir_path.mkdir(parents=True, exist_ok=True)
print("Directory created: ", dir_path)
print(os.path.exists(path=dir_path))

with open(dir_path / str("decomposed_plan.py"), "w") as d:
    d.write(decomposed_plan)

with open(dir_path / str("allocated_plan.py"), "w") as a:
    a.write(allocated_plan)

with open(dir_path / str("code_plan.py"), "w") as c:
    c.write(code_plan)

d.close()
a.close()
c.close()

# STOP - stop the controller / remove window
env.controller.stop()
