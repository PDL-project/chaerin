from pathlib import Path
import os, sys
import argparse
import glob
from misc import extract_number

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--min_agents', type=int)
parser.add_argument('--max_agents', type=int)
parser.add_argument('--seeds', type=int) # seeds up to that amt
args = parser.parse_args()

scenes=glob.glob(str("./Scenes/*"))
scenes=list(map(lambda s : extract_number(Path(s).name), scenes))
scenes=list(filter(lambda v : v is not None, scenes))
scenes.sort()
seeds=[i*10 for i in range(args.seeds)]
print(scenes)

main_s = lambda scn, agnts, seed : f"python baselines/{args.baseline}.py --scene={scn} --name='{args.name}' --agents={agnts} --seed={seed}\n"
pprint  = lambda scn, agnts, seed  : f"python print.py --scene={scn} --agents={agnts} --seed={seed}\n"
bash_directory = f'{args.baseline}.sh'

print(f"file output in '{bash_directory}'")

s=""
for scn in scenes: # scenes are numbers
    for agents in range(args.min_agents, args.max_agents+1):
        print(f'Auto scene_{scn} - {agents} agents ...')
        s+=f"\n # scene_{scn} - {agents} agents - all seeds\n"
        for seed in seeds:
            s+=main_s(scn, agents, seed)
            s+=pprint(scn, agents, seed)

with open(bash_directory, 'w') as f:
    f.write(s)

print('Written.')
