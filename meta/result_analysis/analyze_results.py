"""
Script to calculate confidence intervals for the different metrics
This will be used to generate the tables in the paper
"""

# import environment
from AI2Thor.env_new import AI2ThorEnv
from AI2Thor.base_env import convert_dict_to_string
from AI2Thor.object_actions import get_closest_feasible_action, get_closest_object_id

# import utils for this baseline
from AI2Thor.baselines.llamar.llamar_utils import *
from AI2Thor.baselines.utils.logging import Logger
from AI2Thor.baselines.utils.auto_config import AutoConfig

import argparse
import glob
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="AI2Thor")
parser.add_argument("--baseline", type=str, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agents", type=int, default=2)
parser.add_argument("--granular", action="store_true")
parser.add_argument("--write", type=str, default=None)
parser.add_argument("--read", type=str, default=None)
parser.add_argument("--ci", action="store_true")

# '1_put_bread_lettuce_tomato_fridge'
args = parser.parse_args()

if args.baseline is None:
    bns = glob.glob("./{args.path}/baselines/*")
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

    summary = logger.summarize(num_agents=args.agents)
    task_avg = logger.get_task_average(summary)
    overall_avg = logger.get_overall_average(summary)

    # @change - to task you want to see results for (or None if all tasks)
    # task=None # if None, then do all tasks
    task = args.task

    from pprint import pprint

    if task is not None:
        summary = summary[task]
        task_avg = task_avg[task]

    if args.granular:
        s = f"* Summary of results for '{bn}' baseline *"
        print("*" * len(s))
        print(s)
        print("*" * len(s))
        pprint(summary)
        print()

    s = f"* Task averaged summary of results for '{bn}' baseline *"
    print("*" * len(s))
    print(s)
    print("*" * len(s))
    pprint(task_avg)
    print()

    if task is None:
        s = f"* Overall averaged summary of results for '{bn}' baseline *"
        print("*" * len(s))
        print(s)
        print("*" * len(s))
        pprint(overall_avg)
        print()


from collections import defaultdict
import numpy as np
import pandas as pd
import confidence_intervals as ci

# store the result in a dataframe (with the proper format for confidence intervals)
# TODO: slightly different mean? floating point error?
# only store result in dataframe if it's one baseline (haven't implemented multiple)

if len(bns) == 1 and (args.write or args.ci):
    # summary : {tasks : {metrics : list}}
    # put in order of tasks
    # row x col -> row = metric, col = value list (named 0...n)

    mtrc_d = defaultdict(list)
    for tsk in sorted(list(summary.keys())):
        # don't use summary.items(), it jumbles the dict's internal order
        d = summary[tsk]
        for mtrc, lst in d.items():
            mtrc_d[mtrc] += lst

    cols = sorted(list(mtrc_d.keys()))
    # stack values (metric x list)
    mat = np.array([mtrc_d[mtrc] for mtrc in cols])
    # tranpose to -> (list x metric)
    mat = mat.T

    # change cols to have similar formatting as excel
    # success_rate -> success rate
    cols = list(map(lambda s: s.replace("_", " "), cols))
    # average steps -> steps
    cols = list(map(lambda s: "steps" if s == "average steps" else s, cols))

    df = pd.DataFrame(mat, columns=cols)

    # write as excel file
    if args.write is not None:
        # TODO: prettify w/ merges for different tasks (w/ color)
        df.to_excel(
            f"{args.write}.xlsx",
            sheet_name=f"{args.baseline} results (merged)",
            columns=cols,
        )

    # if read, load xlsx file to df format for consumption
    if args.read is not None:
        df.read_excel(f"{args.read}.xlsx")

    # calculate and output confidence intervals
    if args.ci:
        ci_metrics = ci.get_ci_metrics(df)
        s = f"* Confidence Intervals (mean, low, high) *"
        print("*" * len(s))
        print(s)
        print("*" * len(s))
        pprint(ci_metrics)
        print()
