from env import SAREnv
from baselines.sar_logging import SARLogger
from pprint import pprint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=int, default=1)
parser.add_argument("--name", type=str, default="llamar")
parser.add_argument("--agents", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

env = SAREnv(num_agents=args.agents, scene=args.scene, seed=args.seed)
env.reset()

# logger
logger = SARLogger(env=env, baseline_name=args.name)
print("baseline path (w/ results):", logger.baseline_path)

actions = logger.actions()

for i, action in enumerate(actions):

    input_dict, successes = env.step(action)
    # print(action, successes)
    # print(input_dict, successes)

    if i == 0 or i == len(actions) - 1:
        print(i, input_dict)
        print(action, successes)
        print("subtasks completed", env.checker.subtasks_completed)
        print()
        # break

    # env.render()
