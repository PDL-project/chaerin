# import utils for this baseline
from baselines.sar_logging import SARLogger

import argparse
import glob
import os
from pprint import pprint
from pathlib import Path

# CI
from collections import defaultdict
import numpy as np
import pandas as pd
import confidence_intervals as ci

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=str, default=None)
parser.add_argument("--granular", action="store_true")

parser.add_argument("--write", type=str, default=None)
parser.add_argument("--read", type=str, default=None)
parser.add_argument("--ci", action="store_true")

args = parser.parse_args()

bn=args.baseline
logger=SARLogger(env=None, baseline_name=bn)
print(logger.baseline_path)

# -------------------------------
s=f"* Baseline: '{bn}' *"
print("*"*len(s))
print(s)
print("*"*len(s))
print()

summaries=logger.summarize()
for agents, summary in summaries.items():
    task_avg=logger.get_task_average(summary)
    overall_avg=logger.get_overall_average(summary)

    # @change - to task you want to see results for (or None if all tasks)
    # task=None # if None, then do all tasks

    s=f"* results for '{bn}' baseline ({agents} agents) *"
    print("*"*len(s))
    print(s)
    print("*"*len(s))

    if args.granular:
        s=f"* Summary of results for '{bn}' baseline ({agents} agents) *"
        print("*"*len(s))
        print(s)
        print("*"*len(s))
        pprint(summary)
        print()


    s=f"* Task averaged summary of results for '{bn}' baseline ({agents} agents) *"
    print("*"*len(s))
    print(s)
    print("*"*len(s))
    pprint(task_avg)
    print()


    s=f"* Overall averaged summary of results for '{bn}' baseline ({agents} agents) *"
    print("*"*len(s))
    print(s)
    print("*"*len(s))
    pprint(overall_avg)
    print()

    # store the result in a dataframe (with the proper format for confidence intervals)
    # NOTE: slightly different mean? floating point error?
    # only store result in dataframe if it's one baseline (haven't implemented multiple)

    if (args.write or args.ci):
        # summary : {tasks : {metrics : list}}
        # put in order of tasks
        # row x col -> row = metric, col = value list (named 0...n)

        mtrc_d=defaultdict(list)
        for tsk in sorted(list(summary.keys())):
            # don't use summary.items(), it jumbles the dict's internal order
            d = summary[tsk]
            for mtrc, lst in d.items():
                mtrc_d[mtrc]+=(lst)
        
        cols=sorted(list(mtrc_d.keys()))
        # stack values (metric x list)
        mat=np.array([mtrc_d[mtrc] for mtrc in cols])
        # tranpose to -> (list x metric)
        mat=mat.T
        
        # change cols to have similar formatting as excel
        # success_rate -> success rate
        cols=list(map(lambda s : s.replace("_", " "), cols))
        # average steps -> steps
        cols=list(map(lambda s : "steps" if s=="average steps" else s, cols))

        df=pd.DataFrame(mat, columns=cols)

        # write as excel file
        if args.write is not None:
            # TODO: prettify w/ merges for different tasks (w/ color) 
            df.to_excel(f"{args.write}_{agents}_agents.xlsx", sheet_name=f'{args.baseline} results (merged)', columns=cols)

        # if read, load xlsx file to df format for consumption
        if args.read is not None:
            df.read_excel(f"{args.read}.xlsx")

        # calculate and output confidence intervals
        if args.ci:
            ci_metrics=ci.get_ci_metrics(df)
            s=f"* Confidence Intervals (mean, low, high) ({agents} agents) *"
            print("*"*len(s))
            print(s)
            print("*"*len(s))
            pprint(ci_metrics)
            print()

    print("\n"*3)
