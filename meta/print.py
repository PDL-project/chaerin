import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=int, default=0)
parser.add_argument('--floorplan', type=int, default=0)
args = parser.parse_args()

print('-'*40)
print(f'Checkpoint: Task {args.task} and FloorPlan {args.floorplan}')
print('-'*40)
