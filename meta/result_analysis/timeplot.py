from plot import tsplot
from AI2Thor.base_env import convert_dict_to_string
from AI2Thor.object_actions import get_closest_feasible_action, get_closest_object_id

# import utils for this baseline
from AI2Thor.baselines.llamar.llamar_utils import *
from AI2Thor.baselines.utils.logging import Logger
from AI2Thor.baselines.utils.auto_config import AutoConfig

import argparse
import glob
from pathlib import Path
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=str, default=None)
parser.add_argument("--agents", type=int, default=2)
parser.add_argument("--max_steps", type=int, default=30)
parser.add_argument("--filedir", type=str, default=None)

args = parser.parse_args()

if args.baseline is None:
    bns = glob.glob("./AI2Thor/baselines/*")
    bns = list(map(lambda fn: Path(fn).name, bns))
    bns = list(
        filter(
            lambda f: f not in ["results", "utils", "baseline_utils", "notebooks"], bns
        )
    )
else:
    bns = [args.baseline]

for bn in bns:
    logger = Logger(env=None, baseline_name=bn, summarize_mode=True)
    print(logger.baseline_path)

    s = f"* Baseline: '{bn}' *"
    print("*" * len(s))
    print(s)
    print("*" * len(s))
    print()

    wrap_text = lambda n: " ".join(n.split("_")).capitalize()

    def pdir():
        if args.filedir is None:
            return lambda x: None
        folder = f"plots/{args.filedir}/"
        if not os.path.exists(folder):
            os.mkdir(folder)
        return lambda mtr: f"plots/{args.filedir}/{mtr}_plot.svg"

    pfn = pdir()

    d_mat = logger.summarize_matrix(
        num_agents=args.agents,
        max_steps=args.max_steps,
        zero_initial=True,
        as_dict=True,
    )
    for mtr, mat in d_mat.items():
        # mat -> datapoints x (max_steps+1) - since we append 0
        path = pfn(mtr)
        tsplot(mat, xlabel="steps", ylabel=wrap_text(mtr), save=path, show=True)
