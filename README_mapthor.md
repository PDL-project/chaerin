# MAP-THOR

Environment to execute LLM-based task planners for multiagent robotics

## How to run baselines in "baselines" folder

### Executing a single task in a single floorplan using a specified baseline model:

- `python {relative_path_to_baseline_pyfile/baseline_name}.py --task={task_number} --floorplan={floorplan_number}`
- - NOTE: sometimes `sudo` before the command is needed
- - baseline_name: any baseline_name.py file in the directory
- - task_number: choose from task 0 to 8
- - floorplan_number: choose from floorplan 0 to 4
- example: `python AI2Thor/baselines/CoT/CoT.py --task=0 --floorplan=0`

### Executing multiple tasks for multiple floorplans using a specified baseline model:

- `bash {relative_path_to_baseline_pyfile/baseline_name}.sh`
- - NOTE: sometimes `sudo` before the command is needed
- - baseline_name: any baseline_name.py file in the directory
- - you can customize which floorplans and tasks to execute by commenting/uncommenting in the baseline_name.sh files
