import os, sys
from pathlib import Path
import subprocess
import argparse
import re
import warnings


# to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# no warnings
warnings.filterwarnings("ignore")

# set directory to top folder level to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files

from AI2Thor.smartLLM_env import SmartLLMEnv
from AI2Thor.baselines.utils import Logger, AutoConfig

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=int, default=0)
parser.add_argument("--floorplan", type=int, default=0)
parser.add_argument("--verbose", type=bool, default=False)
args = parser.parse_args()


def compile_aithor_exec_file():
    # dir_path = logger.baseline_path / logger.env.action_dir_path
    ############ dummy env load ############
    print("Loading environment...")
    auto = AutoConfig()
    auto.set_task(args.task)
    auto.set_floorplan(args.floorplan)

    config = auto.config()

    env = SmartLLMEnv(config)
    d = env.reset(task=auto.task_string())
    env.controller.stop()
    print("Environment loaded successfully!")
    ########################################

    # set action directory path
    dir_path = Path(
        str(os.getcwd() + "/AI2Thor/baselines/results/SmartLLM_partial_obs/")
    )
    dir_path = dir_path / env.action_dir_path
    # example: /AI2Thor/baselines/results/SmartLLM_partial_obs/actions/1_put_bread_lettuce_tomato_fridge/FloorPlan1/2
    ai2thor_connect_path = os.getcwd() + "/AI2Thor/baselines/smartllm/ai2thor_connect"

    executable_plan = ""

    # append the imports to the file
    imports = Path(ai2thor_connect_path + "/import_aux_fns.py").read_text()
    executable_plan += imports + "\n"

    # get the arguments
    # autoconfig
    executable_plan += "auto = AutoConfig()\n"
    executable_plan += f"auto.set_task({args.task})\n"
    executable_plan += f"auto.set_floorplan({args.floorplan})\n"
    executable_plan += f"timeout = auto.get_task_timeout()\n"
    executable_plan += f"config = auto.config()\n"

    # append the connector file
    connector_file = Path(ai2thor_connect_path + "/ai2thor_connect.py").read_text()
    executable_plan += connector_file + "\n"

    # append the allocated plan
    allocated_plan = (dir_path / str("code_plan.py")).read_text()
    python_match = re.search(r"```python(.*?)```", allocated_plan, re.DOTALL)
    if python_match:
        allocated_plan = python_match.group(1)
    executable_plan += allocated_plan + "\n"

    # append the task thread termination
    terminate_plan = Path(ai2thor_connect_path + "/end_thread.py").read_text()
    executable_plan += terminate_plan + "\n"

    with open(dir_path / "final_plan.py", "w") as d:
        d.write(executable_plan)

    return str(dir_path / "final_plan.py")


file_path = compile_aithor_exec_file()
print("Saved file at: ", file_path)
# parser = argparse.ArgumentParser()
# parser.add_argument("--command", type=str, required=True)
# args = parser.parse_args()

# expt_name = args.command
# print(expt_name)
# ai_exec_file = compile_aithor_exec_file(expt_name)

# subprocess.run(["python", ai_exec_file])
