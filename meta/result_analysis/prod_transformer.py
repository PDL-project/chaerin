"""
Code to test sentence_transformers
This will ask for input and then return the closest action and object id
You can give something like 'pick up the apple' and it will return the closest action and object id
"""

# import environment
from AI2Thor.env_new import AI2ThorEnv
from AI2Thor.base_env import convert_dict_to_string
from AI2Thor.object_actions import get_closest_feasible_action, get_closest_object_id

# import utils for this baseline
from AI2Thor.baselines.utils import Logger, AutoConfig

# autoconfig
auto = AutoConfig()
auto.set_task(0)
auto.set_floorplan(0)
timeout = auto.get_task_timeout()

# environment initialization
config = auto.config()

env = AI2ThorEnv(config)
d = env.reset(task=auto.task_string())

while True:
    s = input("Give me input:")
    act = get_closest_feasible_action(s)
    print("action", act)
    act = get_closest_object_id(act, env.object_dict, preaction=s)
    print("closest id", act)
    print(act)
