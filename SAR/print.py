import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=int)
parser.add_argument('--agents', type=int)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

print('-'*40)
print(f'Checkpoint: Scene {args.scene} and num_agents {args.agents} w/ seed {args.seed}')
print('-'*40)
