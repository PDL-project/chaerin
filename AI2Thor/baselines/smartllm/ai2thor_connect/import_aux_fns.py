import json, os, re
import pandas as pd
import tqdm
from pathlib import Path
import sys, argparse, warnings

import math
import re
import shutil
import subprocess
import time
import threading
import cv2
import numpy as np
from scipy.spatial import distance
from typing import Tuple
from collections import deque
import random
import os
from glob import glob


# to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# no warnings
warnings.filterwarnings("ignore")

# set directory to top folder level to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files

from AI2Thor.smartLLM_env import SmartLLMEnv

# import utils for this baseline
from AI2Thor.baselines.utils import Logger, AutoConfig


ai2thor_actions = [
    "GoToObject <robot><object>",
    "OpenObject <robot><object>",
    "CloseObject <robot><object>",
    "BreakObject <robot><object>",
    "SliceObject <robot><object>",
    "SwitchOn <robot><object>",
    "SwitchOff <robot><object>",
    "CleanObject <robot><object>",
    "PickupObject <robot><object>",
    "PutObject <robot><object><receptacleObject>",
    "DropHandObject <robot><object>",
    "ThrowObject <robot><object>",
    "PushObject <robot><object>",
    "PullObject <robot><object>",
]
ai2thor_actions = ", ".join(ai2thor_actions)

# ALL SKILLS - INF MASS (robot1,robot2,robot3,robot4)
robot1 = {
    "name": "robot1",
    "skills": [
        "GoToObject",
        "OpenObject",
        "CloseObject",
        "BreakObject",
        "SliceObject",
        "SwitchOn",
        "SwitchOff",
        "PickupObject",
        "PutObject",
        "DropHandObject",
        "ThrowObject",
        "PushObject",
        "PullObject",
    ],
    "mass": 100,
}
robots = [robot1, robot1, robot1, robot1]


def closest_node(node, nodes, no_robot, clost_node_location):
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[(i * 5) + clost_node_location[i]]
        crps.append(nodes[pos_index])
        return crps


def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5
