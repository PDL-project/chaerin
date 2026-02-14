import argparse
from pathlib import Path
import os, sys
from pprint import pprint

# set parent directory to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files

from env import SAREnv
parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str)
args = parser.parse_args()

# show this environment
env=SAREnv(num_agents=6, scene=args.scene, seed=0)
env.reset()
print("subtasks:")
pprint(env.checker.subtasks)
print("coverage:")
pprint(env.checker.coverage)
env.render()
