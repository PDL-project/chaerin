#!/usr/bin/env python3
"""
Auto-generated multi-robot execution code.
Run with: python <this_file> --floor-plan <N>
"""

import argparse
import sys
import os

import sys
from pathlib import Path
PDL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PDL_ROOT))
sys.path.insert(0, str(PDL_ROOT / "scripts"))
sys.path.insert(0, str(PDL_ROOT / "resources"))
# Add scripts folder to path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_path = os.path.join(base_path, 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from MultiRobotExecutor import MultiRobotExecutor, SubtaskPlan

# --- Robot Assignment ---
ASSIGNMENT = {1: 2, 2: 2, 3: 5, 4: 3, 5: 1, 6: 3}  # subtask_id -> robot_id
PARALLEL_GROUPS = {0: [0, 4, 5, 6], 1: [1], 2: [2], 3: [3]}  # group_id -> [subtask_ids]
AGENT_COUNT = 5
SPAWN_POSITIONS = {1: (-2.75, 0.9009997844696045, 2.75), 2: (-1.25, 0.9009996652603149, 1.0), 3: (-1.25, 0.9009997844696045, 3.25), 4: (-3.75, 0.9009997248649597, 1.25), 5: (-2.5, 0.9009997248649597, 1.25)}  # LP에서 결정된 스폰 좌표

# --- Subtask Plans ---
SUBTASK_PLANS = {
    1: {
        'name': 'subtask_01_Move_the_Bread_to_the_CounterTop',
        'robot_id': 2,
        'actions': ['gotoobject robot1 bread (1)', 'pickupobject robot1 bread diningtable (1)', 'gotoobject robot1 countertop (1)', 'drophandobject robot1 bread countertop (1)'],
        'parallel_group': 1,
    },
    2: {
        'name': 'subtask_02_Move_the_Fork_to_the_Drawer',
        'robot_id': 2,
        'actions': ['gotoobject robot1 fork (1)', 'pickupobject robot1 fork diningtable (1)', 'gotoobject robot1 drawer (1)', 'openobject robot1 drawer (1)', 'drophandobject robot1 fork drawer (1)', 'closeobject robot1 drawer (1)'],
        'parallel_group': 2,
    },
    3: {
        'name': 'subtask_03_Move_the_Spoon_to_the_Drawer',
        'robot_id': 5,
        'actions': ['gotoobject robot1 spoon (1)', 'pickupobject robot1 spoon diningtable (1)', 'gotoobject robot1 drawer (1)', 'openobject robot1 drawer (1)', 'drophandobject robot1 spoon drawer (1)', 'closeobject robot1 drawer (1)'],
        'parallel_group': 3,
    },
    4: {
        'name': 'subtask_04_Move_the_Pan_to_the_StoveBurner',
        'robot_id': 3,
        'actions': ['gotoobject robot1 pan (1)', 'pickupobject robot1 pan diningtable (1)', 'gotoobject robot1 stoveburner (1)', 'drophandobject robot1 pan stoveburner (1)'],
        'parallel_group': 0,
    },
    5: {
        'name': 'subtask_05_Move_the_Tomato_to_the_CounterTop',
        'robot_id': 1,
        'actions': ['gotoobject robot1 tomato (1)', 'pickupobject robot1 tomato diningtable (1)', 'gotoobject robot1 countertop (1)', 'drophandobject robot1 tomato countertop (1)'],
        'parallel_group': 0,
    },
    6: {
        'name': 'subtask_06_Move_the_Mug_to_the_CoffeeMachine',
        'robot_id': 3,
        'actions': ['gotoobject robot1 mug (1)', 'pickupobject robot1 mug diningtable (1)', 'gotoobject robot1 coffeemachine (1)', 'drophandobject robot1 mug coffeemachine (1)'],
        'parallel_group': 0,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--floor-plan', type=int, default=1)
    args = parser.parse_args()
    
    executor = MultiRobotExecutor(base_path)
    executor.assignment = ASSIGNMENT
    executor.parallel_groups = PARALLEL_GROUPS
    
    # Reconstruct subtask_plans
    for sid, data in SUBTASK_PLANS.items():
        executor.subtask_plans[sid] = SubtaskPlan(
            subtask_id=sid,
            subtask_name=data['name'],
            robot_id=data['robot_id'],
            actions=data['actions'],
            parallel_group=data['parallel_group'],
        )
    
    # Execute in AI2-THOR
    executor.execute_in_ai2thor(
        floor_plan=args.floor_plan,
        task_description=TASK_DESCRIPTION if 'TASK_DESCRIPTION' in globals() else None,
        agent_count=AGENT_COUNT,
        spawn_positions=SPAWN_POSITIONS,
    )


if __name__ == '__main__':
    main()