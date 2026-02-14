## Implementation of [SmartLLM](https://arxiv.org/pdf/2309.10062) baseline

We base our implementation on their official codebase available [here](https://github.com/SMARTlab-Purdue/SMART-LLM/tree/master)

## How to execute:

* First get `code_plans` for all tasks by running `run_all_tasks.sh`. This will create files in `AI2Thor/baselines/results/SmartLLM_partial_obs/actions/<task_name>`
* The executable code is stored in `code_plan.py`
* Then run `python AI2Thor/baselines/smartllm/execute_plan.py --task=<task_num> --floorplan=<floor_plan_num>`. 
* This will create an executable code with the name `final_plan.py` in the `AI2Thor/baselines/results/SmartLLM_partial_obs/actions/<task_name>` folder. 
* Execute that file to make the robots work in the AI2Thor simulator.