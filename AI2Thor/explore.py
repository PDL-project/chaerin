import sys
import re
import torch
import json, os
import pandas as pd
from pprint import pprint
import os
import sys
import ast
from pathlib import Path
import enum
import base64
import argparse

import clip
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

# set parent directory to address relative imports
directory = Path(os.getcwd()).absolute()
sys.path.append(
    str(directory)
)  # note: no ".parent" addition is needed for python (.py) files
# print(os.getcwd())


from AI2Thor.env_new import AI2ThorEnv
from AI2Thor.base_env import convert_dict_to_string
from thortils.navigation import find_navigation_plan, get_shortest_path_to_object
from thortils import launch_controller
from thortils.utils.visual import GridMapVisualizer
from thortils.vision import thor_object_bboxes
from thortils.vision import projection as pj
from thortils import constants, thor_object_type
from thortils.map3d import Mapper3D
from thortils.agent import thor_camera_pose, thor_agent_pose
from thortils import proper_convert_scene_to_grid_map
from AI2Thor.summarisers.obs_summariser_2 import ObsSummaryLLM
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch.nn.functional as F
import time 


script_dir = Path(__file__).parent.absolute()
tsv_path = script_dir / "objects_actions.tsv"
df = pd.read_csv(tsv_path, sep="\t")


class Actions(enum.Enum):
    RotateRight = "Rotate(Right)"
    RotateLeft = "Rotate(Left)"
    MoveAhead = "Move(Ahead)"
    MoveBack = "Move(Back)"
    MoveLeft = "Move(Left)"
    MoveRight = "Move(Right)"
    Idle = 'Idle'

    def __str__(self):
        return self.value


"""
    Strategies:
    1) Semantic Map:
        - (a) Generate a semantic map (grid map) of environment 
        - (b) Get a frame from the agent[i]'s pov
        - (c) Subtasks that have yet to be complete
        - Prompt the LLM using (a-c) to come up with a series of discrete actions (Rotation, Movements) or an
            object to navigate to


    ---> semantic map strategy isn't as promising since the semantic maps cannot be generated from the perspective 
        of all/multiple agents. (The controller defaults to one of the agents)
        -- the map quality isn't very high quality (cannot/ haven't figured out how to distinguish between objects
            in the grid space), only shows rough large structure good for getting a rough sense of the layout of the
            environment
    

    2) Rotate Agent/ Use Clip:
        - Rotate the agent[i] in 8 directions (pitching up/down, rotating in all 4 directions)
        - Would be helpful to have a method automaatically move the agent to a speified orientation
        - With each orientation that the agent views pass in the image and subtask list to CLIP and compute a reward
        - Rotate or move in the direction of highest reward
            - Rotate to specified orientation:

            - Move forward:
                - If path is blocked, find a path around the object using navigateto (find an object directly infront of the agent)
                - not blocked: just take a step forward


            - Rotate:
                - Rotate the robot to the orientation of highest reward
        
        #Note: with multiple iterations of explore, if the agent is changing pitch then will have to reset agent position
        to look forward directly.


    #Just focus on the task 2:
    - Todo Monday:
        - Incorperate Clip--> Given a series of images (4 rotates) and a subtask compute which orientation 
        gives the best score
        - Object avoidance: if moving forward figure out the logic for navigating around the obstacle. 

        ---> Use try and except block: try moving forward if failure then use  nagivateTo an object in the list of
            objects (decide which object using sentence Transformer) 

        --> Each movement involvement involves a rotation, translation, then a rotation back to default orientation 
            (for resetting)

"""


class ExploreEnv(AI2ThorEnv):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.sentence_transformer_model = SentenceTransformer(Path(__file__).parent.absolute() / "sentence_transformer/finetuned_model")
        # self.embeddings = torch.FloatTensor(self.sentence_transformer_model.encode(task)) # task embedding

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")


    def get_semantic_map(self):
        controller = launch_controller({**constants.CONFIG, **{"scene": self.scene}})

        event = controller.step(action="Pass")
        agent_pose = thor_agent_pose(event, as_tuple=True)

        mapper = Mapper3D(controller)
        mapper.automate(num_stops=20, sep=1.5)
        # mapper.map.visualize()

        grid_map = mapper.get_grid_map(floor_cut=0.1)  # treat bottom 0.1m as floor
        viz = GridMapVisualizer(grid_map=grid_map, res=30)
        img = viz.render()

        gx, gy, gth = grid_map.to_grid_pose(agent_pose[0][0], agent_pose[0][2], agent_pose[1][1])
        img = viz.draw_robot(img, gx, gy, gth, color=(200, 140, 194))
        # viz.show_img(img)

        return img


    def get_obs(self, agent_id):
        out_dict = self.get_action_llm_input()
        agent = self.agent_names[agent_id]
        agent_obs = ast.literal_eval(out_dict[f"{agent}'s observation"])
        return agent_obs
    
    

    def get_action_llm_input(self):
        """
        Returns the input to the subtask LLM
        ### INPUT FORMAT ###
        {{Task: description of the task the robots are supposed to do,
        {agent_name[i]}'s observation: list of objects the {agent_name[0]} is observing}}
        """
        # extract the agent_name's observations based on how many agents there are
        llm_input_feats = ["Task"]
        for i in range(self.num_agents):
            agent_name = self.agent_names[i]
            llm_input_feats.extend([agent_name + "'s observation", ])

        output = dict((k, self.input_dict[k]) for k in llm_input_feats)
        return output 


    def deserialize(self, items):
        return [re.sub(r"_[0-9]+", "", item) for item in items]

    def reward_score(self, observe, task_embedding):
        """To convert actions like RotateLeft to Rotate(Left)"""
        item_embeddings = [ torch.FloatTensor(self.sentence_transformer_model.encode([obs])) for obs in observe]
        scores = [torch.cosine_similarity(task_embedding, act_embed) for act_embed in item_embeddings]

        if scores:
            reward = torch.stack(scores).mean().item()
        else:
            reward = 0 
        return reward
    

    def move_agent(self, agent_id, action, num=-1, firstround=None):
        # if firstround and num == 0:
        #     agent_actions = firstround[:]
        #     agent_actions[agent_id] = action
        #     # print(f'agent_actions: {agent_actions}, agent_id:{agent_id}')
        #     return agent_actions
        num_agents = self.num_agents
        agent_actions = [Actions.Idle.value]*num_agents
        agent_actions[agent_id] = action
        return agent_actions
    

    def nagivateTo(self, object):
        return f'NavigateTo({object})'
    

    def explore(self, actions, subtasks, iterations=1, reward_metric="sentence_transformer", stride=3):

        # perform the action of non-exploring agents first
        first_action = actions[:]
        agent_id = actions.index("Explore")
        first_action[agent_id] = Actions.Idle.value
        self.step(first_action)
        curr_success = self.curr_step_success
        
        embedding = torch.FloatTensor(self.sentence_transformer_model.encode(", ".join(subtasks))) # task embedding

        moveforward = [Actions.MoveAhead.value]*stride

        movements = {
                0: [Actions.MoveAhead.value]+moveforward,
                1: [Actions.RotateRight.value]+moveforward,
                2: [Actions.RotateRight.value]*2 + moveforward,
                3: [Actions.RotateLeft.value] + moveforward
        }

        for i in range(iterations):

            if reward_metric == "clip":
                image_paths = self.sweep_around(agent_id)
                max_direct = self.find_opt_orientation(image_paths, subtasks)
            else:
                rewards =  self.get_rewards_sentence_transform(agent_id, embedding)
                max_direct = rewards.index(max(rewards))
            
            # print(max_direct)
            actions = movements.get(max_direct)
            # print(actions)

            for act in actions:
                # print(f'excuted action: {self.move_agent(agent_id, act)} by {agent_id}')
                self.step(self.move_agent(agent_id, act))
                if not self.event.metadata['lastActionSuccess']:

                    #nagivate to
                    numbered_objs = self.get_obs(agent_id)
                    objs = self.deserialize(numbered_objs)

                    item_embeddings = [torch.FloatTensor(self.sentence_transformer_model.encode([obs])) for obs in objs]
                    scores = [torch.cosine_similarity(embedding, act_embed) for act_embed in item_embeddings]

                    scores_tensor = torch.stack(scores)
                    max_obj_idx = scores_tensor.argmax().item()

                    max_obj = numbered_objs[max_obj_idx]

                    # print(f'navigate to step: {self.move_agent(agent_id, act)} by {agent_id}')
                    self.step(self.move_agent(agent_id, self.nagivateTo(max_obj)))

                    # add a checker on this too
                    break

        coverage = self.checker.get_coverage()
        transport_rate = self.checker.get_transport_rate()
        finished = self.checker.check_success()

        curr_success[agent_id] = self.curr_step_success[agent_id]

        return convert_dict_to_string(self.input_dict), curr_success

        

    def encode_image(image_path:str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    
    def sweep_around(self, agent_id):
        rotations = rotations = [Actions.RotateRight.value]*4
        image_paths = []
        for i, rot in enumerate(rotations):
            path = self.get_frame(agent_id)
            image_paths.append(path)
            # print(f'agent id:{agent_id} and movement: {self.move_agent(agent_id, rot, first_round)}')
            self.step(self.move_agent(agent_id, rot))

        return image_paths

    
    def get_rewards_sentence_transform(self, agent_id, task_embedding): 
        rotations = [Actions.RotateRight.value]*4

        rewards = []
        for i, rot in enumerate(rotations):
            objects_view = self.deserialize(self.get_obs(agent_id))
            reward = self.reward_score(objects_view, task_embedding)
            rewards.append(reward)
            # print(f'{objects_view} | {reward}')
            movement = self.move_agent(agent_id, rot)
            self.step(movement)
        return rewards 
    

    def find_opt_orientation(self, image_paths, subtasks):
        subtask_inputs = self.clip_tokenizer(subtasks, padding=True, return_tensors="pt")
        text_features = self.clip_model.get_text_features(**subtask_inputs)

        images = [Image.open(path) for path in image_paths]

        image_inputs = self.clip_processor(images=images, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**image_inputs)

        image_features_norm = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        sim_scores = torch.matmul(text_features_norm, image_features_norm.t())

        sim_scores_final = sim_scores.sum(0)
        softmax_scores = F.softmax(sim_scores_final, dim=0)
        max_index = softmax_scores.argmax().item()
        # print(softmax_scores)
        return max_index

    

if __name__ == "__main__":

    task_index = 1
    config_dir = str(directory)
    with open(config_dir + "/config.json", 'r') as f:
        config_dict=json.load(f)
    config_arr = config_dict['tasks']
    task = config_arr[task_index]['task_description']

    model = SentenceTransformer(Path(__file__).parent.absolute() / "sentence_transformer/finetuned_model")
    embeddings = torch.FloatTensor(model.encode(task))


    class Config:
        def __init__(self):
            self.num_agents = 2
            self.scene = "FloorPlan201"
            self.scene_name = "FloorPlan201"
            self.model = "gpt-4"
            self.use_langchain = False
            self.use_strict_format = True
            self.use_obs_summariser = False
            self.use_act_summariser = False
            self.use_action_failure = True
            self.use_subtask = True
            self.use_future_message = True
            self.forceAction = False
            self.use_memory = True
            self.use_plan = True
            self.use_separate_memory = False
            self.use_shared_memory = True
            self.temperature = 0.7
    config = Config()
    # env = AI2ThorEnv(config)

    env = ExploreEnv(config)
    d = env.reset(task=task)

    subtasks = [
        "Alice or Bob is currently trying to find the computer, book, and remote control."
        # "locate the book",
        # "locate the sofa",
        # "locate the remotecontrol",
        # "locate the computer"
    ]
    env.explore(['Explore', 'Idle'],subtasks, reward_metric="sentence_transformer")
    # embedding = torch.FloatTensor(exploration.sentence_transformer_model.encode(", ".join(subtasks)))
    # exploration.get_rewards_sentence_transform(1, embedding)

    # env.step(['Idle']*2)



    # image_paths = [f"test_exploration_images/frame_{i}.png" for i in range(4)]
    # out = exploration.find_opt_orientation2(image_paths, subtasks)

    # print(out)
    
    # exploration.explore(0)
    # env.controller.stop()


                                             