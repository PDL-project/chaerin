import json
import os
import argparse
from pathlib import Path
from datetime import datetime

import openai
import ai2thor.controller

ai2thor_actions = [
    "GoToObject <robot><object>",
    "OpenObject <robot><object>",
    "CloseObject <robot><object>",
    "BreakObject <robot><object>",
    "SliceObject <robot><object>",
    "SwitchOn <robot><object>",
    "SwitchOff <robot><object>",
    "CleanObject <robot><object>",
    "PickupObject <robot><object>",
    "PutObject <robot><object><receptacleObject>",
    "DropHandObject <robot><object>",
    "ThrowObject <robot><object>",
    "PushObject <robot><object>",
    "PullObject <robot><object>",
]
ai2thor_actions = ", ".join(ai2thor_actions)

# ALL SKILLS - INF MASS (robot1,robot2,robot3,robot4)
robot1 = {
    "name": "robot1",
    "skills": [
        "GoToObject",
        "OpenObject",
        "CloseObject",
        "BreakObject",
        "SliceObject",
        "SwitchOn",
        "SwitchOff",
        "PickupObject",
        "PutObject",
        "DropHandObject",
        "ThrowObject",
        "PushObject",
        "PullObject",
    ],
    "mass": 100,
}
robots = [robot1, robot1, robot1, robot1]


class SmartLLMAgent:
    def __init__(self, args, env, partial_obs: bool = False):
        self.args = args
        self.env = env
        self.task = self.env.task
        # whether to use partial observation or give global observation about all objects in the environment
        self.partial_obs = partial_obs
        self.num_agents = args.num_agents
        self.gpt_version = "gpt-4-1106-preview"
        self.available_robots = []
        self.data_path = (
            os.getcwd() + "/AI2Thor/baselines/smartllm/data/pythonic_plans/"
        )
        for i in range(self.num_agents):
            robot_dict_copy = robots[i].copy()
            robot_dict_copy["name"] = "robot" + str(i + 1)
            self.available_robots.append(robot_dict_copy)

        with open(os.path.expanduser("~") + "/openai_key.json") as json_file:
            key = json.load(json_file)
            api_key = key["my_openai_api_key"]
        openai.api_key = api_key
        self.all_objects = self.get_ai2_thor_objects()

    def gpt_call(
        self,
        prompt,
        max_tokens=128,
        temperature=0,
        frequency_penalty=0,
    ):

        response = openai.ChatCompletion.create(
            model=self.gpt_version,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
        )
        return response, response["choices"][0]["message"]["content"].strip()

    # Function returns object list with name and properties.
    def convert_to_dict_objprop(self, objs):
        objs_dict = []
        for i, obj in enumerate(objs):
            obj_dict = {"name": obj}
            # obj_dict = {'name': obj , 'mass' : 1.0}
            objs_dict.append(obj_dict)
        return objs_dict

    def get_ai2_thor_objects(self):
        # controller = ai2thor.controller.Controller(scene="FloorPlan" + str(floor_plan_id))
        # get list of objects from POV of agents
        if self.partial_obs:
            combined_obs = []
            for agent_id in range(self.num_agents):
                obs_list = self.env.get_object_in_view(self.env.event, agent_id)
                # _, obs_list = self.env.generate_obs_text(agent_id, prefix="")
                combined_obs += obs_list
            # remove duplicates
            combined_obs = list(set(combined_obs))
            obj = combined_obs
        else:
            obj = list(
                [
                    obj["objectType"]
                    for obj in self.env.controller.last_event.metadata["objects"]
                ]
            )
        # obj_mass = list([obj["mass"] for obj in controller.last_event.metadata["objects"]])
        obj = self.convert_to_dict_objprop(obj)
        return obj

    def get_decompose_module_prompt(self):
        # prepare train decompostion demonstration for ai2thor samples
        prompt = f"from skills import " + ai2thor_actions
        prompt += f"\nimport time"
        prompt += f"\nimport threading"
        # TODO change to visible objects only
        objects_ai = f"\n\nobjects = {self.all_objects}"
        prompt += objects_ai
        # read input train prompts
        decompose_prompt_file = open(self.data_path + "train_task_decompose.py", "r")
        decompose_prompt = decompose_prompt_file.read()
        decompose_prompt_file.close()

        prompt += "\n\n" + decompose_prompt

        prompt = f"{prompt}\n\n# Task Description: {self.task}"
        return prompt

    def get_allocation_module_prompt(self, decomposed_plan):
        prompt = f"from skills import " + ai2thor_actions
        prompt += f"\nimport time"
        prompt += f"\nimport threading"

        allocated_prompt_file = open(
            self.data_path + "train_allocation_solution.py", "r"
        )
        allocated_prompt = allocated_prompt_file.read()
        allocated_prompt_file.close()

        prompt += "\n\n" + allocated_prompt + "\n\n"

        curr_prompt = prompt + decomposed_plan
        curr_prompt += f"\n# TASK ALLOCATION"
        curr_prompt += f"\n# Scenario: There are {len(self.available_robots)} robots available, The task should be performed using the minimum number of robots necessary. Robots should be assigned to subtasks that match its skills and mass capacity. Using your reasoning come up with a solution to satisfy all contraints."
        curr_prompt += f"\n\nrobots = {self.available_robots}"
        objects_ai = f"\n\nobjects = {self.all_objects}"
        curr_prompt += f"\n{objects_ai}"
        curr_prompt += f"\n\n# IMPORTANT: The AI should ensure that the robots assigned to the tasks have all the necessary skills to perform the tasks. IMPORTANT: Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both and allocate robots based on availablitiy. "
        curr_prompt += f"\n# SOLUTION  \n"
        return curr_prompt

    def get_allocation_code(self, decomposed_plan, allocated_plan):
        prompt = f"from skills import " + ai2thor_actions
        prompt += f"\nimport time"
        prompt += f"\nimport threading"
        objects_ai = f"\n\nobjects = {self.all_objects}"
        prompt += objects_ai
        prompt_file = self.data_path + "train_allocation_code.py"
        code_prompt_file = open(prompt_file, "r")
        code_prompt = code_prompt_file.read()
        code_prompt_file.close()

        prompt += "\n\n" + code_prompt + "\n\n"

        curr_prompt = prompt + decomposed_plan
        curr_prompt += f"\n# TASK ALLOCATION"
        curr_prompt += f"\n\nrobots = {self.available_robots}"
        curr_prompt += allocated_plan
        curr_prompt += f"\n# CODE Solution  \n"
        return curr_prompt

    def generate_solution(self):
        """Run all three LLM modules of SmartLLM to generate a solution for the task."""
        print("Generating Decomposition Solution...")
        curr_prompt = self.get_decompose_module_prompt()
        messages = [{"role": "user", "content": curr_prompt}]
        _, decomposed_plan = self.gpt_call(
            messages, max_tokens=1300, frequency_penalty=0.0
        )

        print("Generating Allocation Solution...")
        curr_prompt = self.get_allocation_module_prompt(decomposed_plan)
        messages = [
            {
                "role": "system",
                "content": "You are a Robot Task Allocation Expert. Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both based on your reasoning. In the case of Task Allocation based on Robot Skills alone - First check if robot teams are required. Then Ensure that robot skills or robot team skills match the required skills for the subtask when allocating. Make sure that condition is met. In the case of Task Allocation based on Mass alone - First check if robot teams are required. Then Ensure that robot mass capacity or robot team combined mass capacity is greater than or equal to the mass for the object when allocating. Make sure that condition is met. In both the Task Task Allocation based on Mass alone and Task Allocation based on Skill alone, if there are multiple options for allocation, pick the best available option by reasoning to the best of your ability.",
            },
            {"role": "system", "content": "You are a Robot Task Allocation Expert"},
            {"role": "user", "content": curr_prompt},
        ]
        _, allocated_plan = self.gpt_call(
            messages, max_tokens=400, frequency_penalty=0.69
        )

        print("Generating Allocated Code...")
        curr_prompt = self.get_allocation_code(decomposed_plan, allocated_plan)
        messages = [
            {"role": "system", "content": "You are a Robot Task Allocation Expert"},
            {"role": "user", "content": curr_prompt},
        ]

        _, code_plan = self.gpt_call(messages, max_tokens=1400, frequency_penalty=0.4)
        return decomposed_plan, allocated_plan, code_plan


# homogenous robots; prepare list of robots for the tasks
# robots = [robot1, robot1, robot1, robot1]
# based on number of robots, create available_robots
# num_agents = 2
# available_robots = []
# for i in range(num_agents):
#     # rename the robot
#     robot_dict_copy = robots[i].copy()
#     robot_dict_copy["name"] = "robot" + str(i + 1)
#     print(robot_dict_copy)
#     available_robots.append(robot_dict_copy)

# print("Generating Decomposition Solution...")
# curr_prompt = get_decompose_module_prompt(test_tasks[0], args.floor_plan)
# # use only gpt4 because SmartLLM shows better performance with gpt4
# messages = [{"role": "user", "content": curr_prompt}]
# _, decomposed_plan = LM(messages, "gpt-4", max_tokens=1300, frequency_penalty=0.0)

# print("Generating Allocation Solution...")
# curr_prompt = get_allocation_module_prompt(
#     floor_plan, decomposed_plan, available_robots
# )
# # gpt 4.0
# messages = [
#     {
#         "role": "system",
#         "content": "You are a Robot Task Allocation Expert. Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both based on your reasoning. In the case of Task Allocation based on Robot Skills alone - First check if robot teams are required. Then Ensure that robot skills or robot team skills match the required skills for the subtask when allocating. Make sure that condition is met. In the case of Task Allocation based on Mass alone - First check if robot teams are required. Then Ensure that robot mass capacity or robot team combined mass capacity is greater than or equal to the mass for the object when allocating. Make sure that condition is met. In both the Task Task Allocation based on Mass alone and Task Allocation based on Skill alone, if there are multiple options for allocation, pick the best available option by reasoning to the best of your ability.",
#     },
#     {"role": "system", "content": "You are a Robot Task Allocation Expert"},
#     {"role": "user", "content": curr_prompt},
# ]
# _, allocated_plan = LM(
#     messages, "gpt-4-1106-preview", max_tokens=400, frequency_penalty=0.69
# )


# print("Generating Allocated Code...")

# curr_prompt = get_allocation_code(
#     floor_plan, decomposed_plan, allocated_plan, available_robots
# )

# # using variants of gpt 4 or 3.5
# messages = [
#     {"role": "system", "content": "You are a Robot Task Allocation Expert"},
#     {"role": "user", "content": curr_prompt},
# ]
# _, code_plan = LM(
#     messages, "gpt-4-1106-preview", max_tokens=1400, frequency_penalty=0.4
# )


# save generated plan
# exec_folders = []
# if args.log_results:
#     line = {}
#     now = datetime.now()  # current date and time
#     date_time = now.strftime("%m-%d-%Y-%H-%M-%S")

#     for idx, task in enumerate(test_tasks):
#         task_name = "{fxn}".format(fxn="_".join(task.split(" ")))
#         task_name = task_name.replace("\n", "")
#         folder_name = f"{task_name}_plans_{date_time}"
#         exec_folders.append(folder_name)

#         os.mkdir("./logs/" + folder_name)

#         with open(f"./logs/{folder_name}/log.txt", "w") as f:
#             f.write(task)
#             f.write(f"\n\nGPT Version: {args.gpt_version}")
#             f.write(f"\n\nFloor Plan: {args.floor_plan}")
#             f.write(f"\n{objects_ai}")
#             f.write(f"\nrobots = {available_robots[idx]}")
#             f.write(f"\nground_truth = {gt_test_tasks[idx]}")
#             f.write(f"\ntrans = {trans_cnt_tasks[idx]}")
#             f.write(f"\nmax_trans = {max_trans_cnt_tasks[idx]}")

#         with open(f"./logs/{folder_name}/decomposed_plan.py", "w") as d:
#             d.write(decomposed_plan[idx])

#         with open(f"./logs/{folder_name}/allocated_plan.py", "w") as a:
#             a.write(allocated_plan[idx])

#         with open(f"./logs/{folder_name}/code_plan.py", "w") as x:
#             x.write(code_plan[idx])
