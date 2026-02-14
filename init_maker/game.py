import os
import time
import warnings
from pathlib import Path
import sys
import math
import argparse 
import matplotlib.pyplot as plt
import json


# set parent directory to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files

# ------ argparsing / pre-inits ------
parser=argparse.ArgumentParser()
parser.add_argument('--json', type=str, default=None, help="Name of the json file to use for the creation of the pre-init")
parser.add_argument('--floorplan', type=str, default=None, help="The floorplan to initialize this to")
args=parser.parse_args()

assert args.floorplan is not None, 'No floorplan provided, please provide via --floorplan="FloorPlanName" argument'

if args.json is None:
    raise ValueError("Need json file name to run this file and create pre-inits!")

with open(f'init_maker_easy/{args.json}.json') as f:
    metadata = json.load(f)
    # add floorplan information here
    metadata["floorplan"]=args.floorplan

    # example
    #    {
    #    "task_folder" : "1_open_all_cabinets",
    #    "task_folder" : "1_close_all_cabinets",
    #    "task_type" : "manipulation",
    #    "task_complexity" : "simple",
    #    "no_movement" : false,
    #    "actions" : {"OpenObject" : ["cabinets"]},
    #    "docs" : "Open all cabinets in the pre-initialization"
    #    }

# ------ ai2thor ------ 
from AI2Thor.env_new import AI2ThorEnv
from AI2Thor.baselines.utils import AutoConfig

# to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore")

# config
auto = AutoConfig()
auto.set_task(0)
auto.set_floorplan(0)
config = auto.config()
config.scene=metadata["floorplan"]
config.scene_name=metadata["floorplan"]

env = AI2ThorEnv(config)
env.verbose=False
env.skip_save_dir=True
d = env.reset(task=auto.task_string(), ignore_applicable=True)

# -------- py game ------
import pygame
from init_maker_easy.ui import *

# set env from above
UI.env=env

# handles information for initialization
inith = InitHandler(metadata)

pygame.init()
m = Menu(inith)
m.run()
pygame.quit()

# quit AI2Thor
env.controller.stop()
