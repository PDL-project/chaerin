import core
import os, cv2, copy
from core import Controller
from utils import render
from misc import read_enum
from copy import deepcopy
from base_env import SARBaseEnv
from typing import List, Tuple, Dict
from pathlib import Path
from Scenes.get_scene_init import get_scene_initializer

""" Anything method doesn't require controller instance is in SARBaseEnv """

class SAREnv(SARBaseEnv):
    """
    Search and Rescue environment,
    has same interface as AI2Thor env
    thus allowing for exact same use in the baseline files.
    """

    """
    Controller docs:

    ------------------------------
    .step(action, action_args)
        Formatting for actions, required action_args:

        'NavigateTo' : to_target_id (location),
        'Move' : direction, # Up, Down, Left, Right
        'Explore',
        'Carry' : from_target_id (person),
        'DropOff' : to_target_id (deposit), 
        'StoreSupply' : to_target_id (deposit),
        'UseSupply' : from_target_id (fire), supply_type (on fire)
        'GetSupply' : from_target_id (deposit or reservoir), supply_type (deposit)
        'ClearInventory',
        'NoOp',
    ------------------------------

    ------------------------------
    event={
        'success' : None,
        'global_obs' : None,
        'local_obs' : None,
        'visual_obs' : None, # non-existent for now
        'info' : '',
        }

        global_obs -> unordered list of formatted_objects
        local_obs -> dict. key<->(cardinal direction), value<->(list of formatted_objects @ delta)
            cardinal directions ->(Up,Down),(Left,Right) and all pairs from group 1 and 2
    ------------------------------

    ------------------------------
    Format of formatted_objects
        {
        # FOR ALL
        'position' : x,y position of the object,
        'name' : given name for object,
        'object_id' : id of the object,
        'type' : type of object (class)
        'collidable' : is object collidable,

        # SPECIFIC
        (if flammable) 'intensity' :  'none', 'low', 'medium', 'high',
        (if flammable) 'fire_type' : 'chemical' or 'non-chemical',

        (if fire) 'average_intensity' : 'none', 'low', 'medium', 'high',
        (if fire) 'fire_type' : 'chemical' or 'non-chemical',

        (if person) 'load' : number corresponding to amt of agents needed,
        (if person) 'status' : 'grabbed' or 'grounded',

        (if reservoir) 'resource_type' : 'sand' or 'water',
        (if reservoir) 'inventory' : dict w/ {resource_type : amt} (likely math.inf),

        (if deposit) 'inventory' : dict w/ {resource_type : amt} for all resource types (including person),

        (if absagent) 'inventory' : dict w/ {resource_type : amt} for all resource types (including person),
        }
    ------------------------------
    """
    AGENT_NAMES=['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Finn']
    AVAILABLE_ACTIONS=Controller.ALL_ACTIONS+['Explore']
    CARDINAL_DIRECTIONS=Controller.MOVABLE_CARDINAL_DIRECTIONS

    # To be used in prompt (describing environment), useful variables
    CLASS_TYPES=core.Field.CLASS_TYPES[:-1] # exclude AbsAgent

    INVENTORY_CAPACITY=core.AbsAgent.INVENTORY_CAPACITY
    INVENTORY_TYPES=list([k.capitalize() for k in core.Field.UNREADABLE_TYPE_MAPPER_RESOURCE.keys()])
    FIRE_TYPES=list([k.capitalize() for k in core.Field.UNREADABLE_TYPE_MAPPER_FIRE.keys()])
    EXTINGUISH_TYPES=INVENTORY_TYPES[:len(FIRE_TYPES)] # assumed to be first n -> fire extinguish types
    MIN_REQUIRED_AGENTS=core.Person.MIN_REQUIRED_AGENTS

    L_TO_M=core.Flammable.L_TO_M
    M_TO_H=core.Flammable.M_TO_H
    CRITICAL_INTENSITY=read_enum(core.Flammable.CRITICAL_INTENSITY)
    AMT_INTENSITIES=len(list(filter(lambda s : '_' not in s, dir(core.Intensity))))
    ALL_INTENSITIES=[read_enum(core.Intensity(i)) for i in range(1, AMT_INTENSITIES+1)]
    GRID_HEIGHT=core.Coordinate.HEIGHT
    GRID_WIDTH=core.Coordinate.WIDTH


    def __init__(self, num_agents, scene=1, seed=42, save_frames=False):
        # NOTE: AutoConfig NOT needed
        self.seed=seed
        SARBaseEnv.seed=self.seed
        self.num_agents=num_agents
        self.scene=scene
        self.save_frames=save_frames

        assert 1<=num_agents<=len(SAREnv.AGENT_NAMES), f"Amount of agents <1 or >maximum."
        self.agent_names=SAREnv.AGENT_NAMES[:num_agents]
        # only true when .reset()
        self.initialized=False


    def reset(self):
        """ Resets and initializes the environment (controller) """

        # indication that we can do .step(.)
        self.initialized=True

        scene_initializer,checker=get_scene_initializer(self.scene)

        scene_initializer_instance=scene_initializer.SceneInitializer()
        self.task_timeout=scene_initializer_instance.task_timeout
        self.task=scene_initializer_instance.get_task()

        self.controller, pg_params=scene_initializer_instance.preinit(
            self.num_agents, self.agent_names, self.seed
        )
        self.object_dict = pg_params # used in obj actions & init-ing checker
        self.checker = checker.Checker(copy.deepcopy(self.object_dict))

        self.input_dict = {}
        self.input_dict["Task"] = self.task

        self.open_subtasks=[]
        self.closed_subtasks=[]
        self.subtasks=[[] for _ in range(self.num_agents)]
        self.memory=[[] for _ in range(self.num_agents)]

        self.step_num = [0] * self.num_agents

        self.all_obs_dict = { self.agent_names[i]: [] for i in range(self.num_agents) }
        self.action_history = { self.agent_names[i]: [] for i in range(self.num_agents) }
        self.action_success_history = { self.agent_names[i]: [] for i in range(self.num_agents) }
        self.agent_failure_acts = { self.agent_names[i]: set([]) for i in range(self.num_agents) } # set since we don't want to be redundant
        self.step_nums_history = { self.agent_names[i]: [] for i in range(self.num_agents) }
        self.previous_success = { self.agent_names[i]: [] for i in range(self.num_agents) }

        self.update_current_state(["I have not taken any actions yet"]*self.num_agents)

        base_path=f"scene_{self.scene}/"
        self.render_image_path=Path(f"render/{self.num_agents}_agents/seed_{self.seed}/{base_path}")
        self.action_dir_path=Path(f"actions/{self.num_agents}_agents/seed_{self.seed}/{base_path}")

        return SARBaseEnv.convert_dict_to_string(self.input_dict)

    def update_current_state(self, act_texts: List[str]):
        """ Update the input dictionary with the current state of the environment """
        # NOTE: Since we get observation after we do all the steps, we'll have non-stale observations

        for agent_idx in range(self.num_agents):
            agent_name = self.agent_names[agent_idx]

            obs_text,global_obs_list = self.generate_obs_text(agent_idx)
            state=self.get_agent_state(agent_idx)  # textual description of the agent's state (heading, position, inventory, etc)
            act_failure_text=self.get_act_failure_text(self.agent_failure_acts[agent_name], agent_idx)

            # --- update the input dict ---

            self.input_dict[agent_name + "'s observation"] = obs_text
            self.input_dict[agent_name + "'s state"] = state
            # MISSING: if use_act_summariser, make [agent_name + "'s feasible actions"] <- get_feasible_actions(global_obs_list, agent_idx)
            self.input_dict[agent_name + "'s previous observation"] = self.all_obs_dict[agent_name]
            self.input_dict[agent_name + "'s previous action"] = act_texts[agent_idx]
            self.input_dict[agent_name + "'s previous failures"]=act_failure_text

            self.all_obs_dict[self.agent_names[agent_idx]]=global_obs_list # add previous history
            
        # MISSING: separate subtasks
        # MISSING: separate memories
        # default to shared for both of these
        self.input_dict["Robots' subtasks"]=self.subtasks[0]
        self.input_dict["Robots' combined memory"]=self.memory[0]
        self.input_dict["Robots' open subtasks"]=self.open_subtasks
        self.input_dict["Robots' completed subtasks"]=self.closed_subtasks

    def generate_obs_text(self, agent_idx):
        dct=self.controller.get_observation(agent_idx)
        lobs=dct['local_obs']
        gobs=dct['global_obs']

        # --- preprocessing ---
        # lobs is delta -> obj dict list
        # change to delta -> str of string_description (if flammable & not blocked) else 'obstacle' list
        for direction,ldcts in lobs.items():
            def is_flammable(d):
                if d['type']=='Flammable':
                    return f"Flammable of fire (intensity {d['intensity']}): {d['parent_fire']}"
                return 'Obstacle'
            ldcts=list(map(is_flammable, ldcts))
            
            v=ldcts # value to replace lobs[direction] dict with
            # we're assuming abstract objects such as fire are not part of lobs (thus no need for filter)

            if ('Obstacle' in ldcts): v=['Obstacle']
            elif len(ldcts)==0: v=['Empty']

            lobs[direction]=str(v)

        # if a direction isn't in lobs, it's a wall (obstacle)
        for direction in Controller.MOVABLE_CARDINAL_DIRECTIONS:
            if direction not in lobs.keys():
                lobs[direction]=['Obstacle']

        # make sure that dict has same order
        lobs={direction: lobs[direction] for direction in Controller.MOVABLE_CARDINAL_DIRECTIONS}
        # -------------------

        # gobs is list of obj dicts, replace w/ readable strings
        readable=lambda d : d['string_description']
        get_names=lambda d: d['name']

        # --- preprocessing ---
        # list of names only w/o description in global obs
        gobs_names=list(map(get_names, gobs))
        gobs=list(map(readable, gobs))

        # @here
        # NOTE: Filter out fire from the observation here - to avoid having the agent 
        #       have center (could have double fire) / non-center (redundant) positions to the fire (since we already have regions)
        gobs_names=list(filter(lambda s : ('Fire' not in s) or ('_Region' in s), gobs_names))


        # --- text ---
        # add local
        str_lobs=SARBaseEnv.convert_dict_to_string(lobs, pre='\t')
        obs_text=f"\n\tDirectly around me, I can see:\n\t{str_lobs}," # {'Left' : ['Obstacle'], 'Right' : ['Empty'], 'UpRight' : ['Flammable fire: GreatFire'], ...}
        # add global
        obs_text+=f"\n\tGlobally, I can see: {str(gobs)}," # ['Flammable object named GreatFire_Region_1 with an intensity of Low', ...]
        # add global names
        obs_text+=f"\n\tNames: {str(gobs_names)}" # ['GreatFire', 'LostTimmy']

        # info to store for previous observation
        global_obs_list=str(gobs_names)

        return obs_text,global_obs_list

    def get_agent_state(self, agent_idx: int):
        """
        Return a string describing the agent's state
        """
        agent=self.controller.get('agents', agent_idx)
        inventory=self.controller.get_inventory(agent_idx)
        agent_state = f"I am at co-ordinates: {agent.get_position()} and I am holding {inventory}."
        return agent_state

    def update_memory(self, memory : list, agent_idx : str):
        self.memory[agent_idx]=memory

    def update_subtask(self, subtask : list, agent_idx : str):
        self.subtasks[agent_idx]=subtask

    def update_plan(self, plan : str):
        self.plan=plan

    def get_id(self, name : str):
        # could be None if name doesn't exist
        return self.controller.get_id(name)

    # ------- prompts -------
    def get_planner_llm_input(self):
        """
        Returns the input to the subtask LLM
        ### INPUT FORMAT ###
        {{Task: description of the task the robots are supposed to do,
        {agent_name[0]}'s observation: list of objects the {agent_name[0]} is observing,
        {agent_name[1]}'s observation: list of objects the {agent_name[1]} is observing,
        Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
        Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
        }}

        """
        # extract the agent_name's observations based on how many agents there are
        planner_llm__input_feats = ["Task"]
        for i in range(self.num_agents):
            agent_name = self.agent_names[i]
            planner_llm__input_feats.append(agent_name + "'s observation")
        planner_llm__input_feats.extend(
            ["Robots' open subtasks", "Robots' completed subtasks"]
        )
        return dict((k, self.input_dict[k]) for k in planner_llm__input_feats)

    def get_verifier_llm_input(self):
        """
        Returns the input to the verifier LLM
        ### INPUT FORMAT ###
        {{Task: description of the task the robots are supposed to do,
        {agent_name[i]}'s observation: list of objects the {agent_name[0]} is observing,
        {agent_name[i]}'s previous action: previous action of the {agent_name[0]},
        Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
        Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
        Robots' combined memory: description of robots' combined memory}}

        """
        # extract the agent_name's observations based on how many agents there are
        verifier_llm_input_feats = ["Task"]
        for i in range(self.num_agents):
            agent_name = self.agent_names[i]
            verifier_llm_input_feats.extend(
                [
                    agent_name + "'s observation",
                    agent_name + "'s state",
                    agent_name + "'s previous action",
                ]
            )
        verifier_llm_input_feats.extend(
            [
                "Robots' open subtasks",
                "Robots' completed subtasks",
                "Robots' combined memory",
            ]
        )
        return dict((k, self.input_dict[k]) for k in verifier_llm_input_feats)

    def get_action_llm_input(self, failure_module=False):
        """
        Returns the input to the subtask LLM
        ### INPUT FORMAT ###
        {{Task: description of the task the robots are supposed to do,
        {agent_name[i]}'s observation: list of objects the {agent_name[0]} is observing,
        {agent_name[i]}'s state: description of {agent_name[0]}'s state,
        {agent_name[i]}'s previous action: description of what {agent_name[0]} did in the previous time step and whether it was successful,
        {agent_name[i]}'s previous failures: if {agent_name[0]}'s few previous actions failed, description of what failed,
        Robots' open subtasks: list of subtasks  supposed to carry out to finish the task. If no plan has been already created, this will be None.
        Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
        Robots' subtask: description of the subtasks the robots were trying to complete in the previous step,
        Robots' combined memory: description of robot's combined memory}}
        """
        llm_input_feats = ["Task"]
        for i in range(self.num_agents):
            agent_name = self.agent_names[i]
            llm_input_feats.extend(
                [
                    agent_name + "'s observation",
                    agent_name + "'s state",
                    agent_name + "'s previous action",
                    agent_name + "'s previous failures",
                ]
            )
        llm_input_feats.extend(
            [
                "Robots' open subtasks",
                "Robots' completed subtasks",
                "Robots' combined memory",
            ]
        )

        if failure_module:
            # post action / failure module inputs
            # v0 - give failure reason (override environment-given), add logic for next action
            llm_input_feats.extend(
                    [
                        "failure reason",
                        # "logic for next action",
                    ]
            )
        return dict((k, self.input_dict[k]) for k in llm_input_feats)

    def step(self, actions):
        """
        Execute the actions for all the agents in the environment
        Return the observation for all the agents
        """
        assert self.initialized, f"Environment not .reset(); can't do .step(.)"

        act_successes, act_texts = [], []

        # NOTE: here we append the successes from the drop action
        #       at the end, we have to assign True to success of ALL agents that did this action if any of these is true
        drop_successes={}

        for agent_idx in range(self.num_agents):
            action=actions[agent_idx]

            # @ coela - added this to handle SendMessage
            if "SendMessage" in action:
                act_success=True
                error_type=''
            else:
                action_kwargs=self.parse_action(action, agent_idx)

                # multiple steps at once, used in Explore
                if "Explore" in action:
                    events=[self.controller.step(**kwargs) for kwargs in action_kwargs]
                    # since observations are the same, use any event that was successful (if such exists)
                    successes=[int(e['success']) for e in events]
                    if 1 in successes: self.event=events[successes.index(1)]
                    else: self.event=events[-1]

                else: self.event=self.controller.step(**action_kwargs)

                act_success=self.event['success']
                error_type=self.event['error_type']

                self.previous_success[agent_idx]=act_success
                self.step_num[agent_idx]+=1

            # save the current frame
            if self.save_frames: self.save_frame()

            # @bug in original implementation, failure acts don't say previous failures
            # fixed by not resetting if successful
            if not act_success:
                self.agent_failure_acts[self.agent_names[agent_idx]].add(action)
            else:
                # on the flip side, remove if this action has been successful (otherwise the agent will get confused)
                if action in self.agent_failure_acts[self.agent_names[agent_idx]]:
                    self.agent_failure_acts[self.agent_names[agent_idx]].remove(action)

            # add the action and succes to the action and success history
            self.action_history[self.agent_names[agent_idx]].append(action)
            self.step_nums_history[self.agent_names[agent_idx]].append(self.step_num[agent_idx])
            self.action_success_history[self.agent_names[agent_idx]].append(act_success)

            # NOTE: that the this observation ISN'T stale, so checker can properly check for ending fire (once last agent has finished)
            # @here
            self.checker.perform_metric_check(
                action, act_success, self.controller.get_observation(agent_idx)
            )

            act_text = self.get_act_text(
                action, act_success, agent_idx, error_type
            )

            # add drop off success
            if "DropOff" in action:
                drop_successes[agent_idx]=act_success

            act_texts.append(act_text)
            act_successes.append(act_success)

        # change successes for DropOff action
        if any(drop_successes.values()):
            # update all the previous information to successful
            for idx in drop_successes.keys():
                act=self.action_history[self.agent_names[idx]][-1]

                self.previous_success[idx]=True
                act_successes[idx]=True
                act_texts[idx]=self.get_act_text(act, True, idx, '')

                self.action_success_history[self.agent_names[idx]][-1]=True

                # if successful, also remove from the failure acts
                if act in self.agent_failure_acts[self.agent_names[idx]]:
                    self.agent_failure_acts[self.agent_names[idx]].remove(act)

        self.update_current_state(act_texts)
        return SARBaseEnv.convert_dict_to_string(self.input_dict), act_successes

    def render(self, save_path=None, show=True):
        # cute render function
        render(self.controller.field.all_objects(expand=True, with_memory=False), save_path=save_path, show=show)

    def _create_path(self, pth):
        if not os.path.exists(pth):
            os.makedirs(pth)

    def save_frame(self):
        agent_dir_pov = self.render_image_path
        self._create_path(agent_dir_pov)

        pth = agent_dir_pov / f"frame_{self.step_num[0]}.png"
        self.render(save_path=pth, show=False)


if __name__=="__main__":
    env=SAREnv(num_agents=3)
