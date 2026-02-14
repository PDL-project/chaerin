from pathlib import Path
import os, sys

# set parent directory to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files

from AI2Thor.baselines.utils.logging import Logger
from AI2Thor.baselines.utils.auto_config import AutoConfig

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=str)
parser.add_argument("--task", type=int, default=None)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--agents", type=int, default=2)
parser.add_argument("--config_file", type=str, default="config.json")
args = parser.parse_args()

# autoconfig
auto = AutoConfig(config_file=args.config_file)
num_tasks = auto.get_amt_tasks()


# main_s = lambda t,fp : f"sudo python AI2Thor/baselines/llamar/llamar.py --task={t} --floorplan={fp}\n"
# pprint  = lambda t,fp  : f"sudo python print.py --task={t} --floorplan={fp}\n"
# bash_directory = 'llamar.sh'

assert (
    args.name is not None
), f"Must provide --name argument (this is the name of the folder to put all results)"
print(f"Creating bash file for '{args.baseline}' baseline\n")

# if args.name is None:
# args.name=args.baseline
# @change - main_s assumed that the folder of the baseline is the same as the name of the baseline.py file!!
main_s = (
    lambda t, fp: f"sudo python AI2Thor/baselines/{args.baseline}/{args.baseline}.py --task={t} --floorplan={fp} --name='{args.name}' --agents={args.agents} --config_files={args.config_file}\n"
)
pprint = lambda t, fp: f"sudo python meta/print.py --task={t} --floorplan={fp}\n"
bash_directory = f"meta/{args.baseline}.sh"
print(f"file output in '{bash_directory}'")

s = ""
tasks = [args.task] if args.task is not None else range(num_tasks)
for t in tasks:
    print(f"Auto {t}...")
    auto.set_task(t)
    num_fps = auto.get_amt_floorplans(t)
    s += f"\n # task {t} - all floorplans \n"
    for fp in range(num_fps):
        s += main_s(t, fp)
        s += pprint(t, fp)

with open(bash_directory, "w") as f:
    f.write(s)

print("Written.")
