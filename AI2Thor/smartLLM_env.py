import os, sys
import json
import cv2
import argparse
from typing import List, Tuple, Dict
import ai2thor.controller
from ai2thor.platform import CloudRendering
import openai

# save a json file with your openai api key in your
# home folder as {"my_openai_api_key": "INSERT API HERE"}
with open(os.path.expanduser("~") + "/openai_key.json") as json_file:
    key = json.load(json_file)
    openai_api_key = key["my_openai_api_key"]
openai.api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
sys.path.append(os.path.abspath(os.getcwd()))

from AI2Thor.base_env import convert_dict_to_string, BaseEnv
from AI2Thor.object_actions import get_interactable_objects
from AI2Thor.Tasks.task_mapper import get_closest_task
from AI2Thor.Tasks.get_scene_init import get_scene_initializer


class SmartLLMEnv(BaseEnv):
    """Environment compatible with SmartLLM baseline official implementation"""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.scene = args.scene
        self.num_agents = args.num_agents
        self.controller = ai2thor.controller.Controller(
            width=1000,
            height=1000,
            scene=args.scene,
            gridSize=0.25,
            # platform=CloudRendering,
        )
        self.agent_names = ["Alice", "Bob", "Charlie", "Dirk"]
        self.move_actions = ["MoveAhead", "MoveBack", "MoveRight", "MoveLeft"]
        self.rotate_actions = ["RotateRight", "RotateLeft"]
        self.obs_summary_llm_cache_path = "AI2Thor/summary_llm_cache.pkl"
        # self.model = "gpt-4-0314"
        self.model = args.model
        self.use_plan = args.use_plan if hasattr(args, "use_plan") else False
        self.forceAction = args.forceAction
        self.previous_actions = [None] * self.num_agents  # to store previous actions
        self.previous_success = [None] * self.num_agents  # to store previous success
        self.inventory = [
            "nothing"
        ] * self.num_agents  # to store what objects the agent is holding
        # to store previous observations
        self.all_obs = []
        self.agent_failure_acts = {
            self.agent_names[i]: [] for i in range(self.num_agents)
        }

        # overhead vs agent-centric
        self.overhead = False

    def reset(
        self,
        task: str = "bring a tomato, lettuce and bread to the countertop to make a sandwich",
    ):
        """
        input_dict = {
            Task: <task>,
            Alice's observation: <observation>,
            Alice's state: <state>,
            Alice's feasible actions: <feasible actions>,
            Alice's previous observation: <previous observation>,
            Alice's previous action: <previous action>,
            Bob's observation: <observation>,
            Bob's state: <state>,
            Bob's feasible actions: <feasible actions>,
            Bob's previous observation: <previous observation>,
            Bob's previous action: <previous action>,
        }
        """
        self.visibilityDistance = (
            1.5  # set this so we can use it in navigation later on
        )
        self.event = self.controller.step(
            dict(
                action="Initialize",
                gridSize=0.25,
                renderObjectImage=True,
                agentCount=self.num_agents,
                visibilityDistance=self.visibilityDistance,
            )
        )
        self.object_dict = {}
        self.task = task
        self.input_dict = {}
        self.input_dict["Task"] = task
        self.task, self.closest_task_instr = get_closest_task(task)
        # TODO: remove for me
        print("_" * 50)
        print(f"Pre-initializing environment with closest task: {self.task}")
        print("_" * 50)
        self.create_save_dirs()

        self.pre_sum_obs_list = {
            self.agent_names[i]: [] for i in range(self.num_agents)
        }

        # move agent-0 to an initial position away from agent-1
        self.agent_init_pos(0)
        self.step_num = [0] * self.num_agents
        self.all_obs = []

        # @UROPs this is where preinit will be called
        scene_initializer, checker = get_scene_initializer(self.task, self.scene)
        self.checker = checker.Checker()
        # TODO
        print("_" * 50)
        print("Subtasks to complete:")
        print("\n".join(self.checker.subtasks))

        if scene_initializer is not None:
            # TODO
            print("_" * 50)
            print("Preinitializing the environment")
            print("_" * 50)

            # @sid - created instance of class because preinit isn't static method (gave error)
            self.event = scene_initializer.SceneInitializer().preinit(
                self.event, self.controller
            )

        #######################################
        # make a dict of lists for action history for each agent
        self.action_history = {self.agent_names[i]: [] for i in range(self.num_agents)}
        self.action_success_history = {
            self.agent_names[i]: [] for i in range(self.num_agents)
        }
        self.step_nums_history = {
            self.agent_names[i]: [] for i in range(self.num_agents)
        }
        self.all_obs_dict = {self.agent_names[i]: [] for i in range(self.num_agents)}

        for agent_id in range(self.num_agents):
            # TODO - add back frame
            self.save_frame(agent_id)
            agent_name = self.agent_names[agent_id]
            obs, obs_list = self.generate_obs_text(agent_id, prefix="")
            self.input_dict[agent_name + "'s observation"] = obs
            state = self.get_agent_state(agent_id)
            self.input_dict[agent_name + "'s state"] = state
            # obs_list = self.convert_str2list(obs)
            self.input_dict[agent_name + "'s previous observation"] = "None"
            self.input_dict[agent_name + "'s previous action"] = (
                "I have not taken any action yet"
            )
            self.all_obs.append(obs_list)
            self.all_obs_dict[agent_name] = obs_list

        return convert_dict_to_string(self.input_dict)

    def get_reachable_positions(self):
        reachable_positions_ = self.controller.step(
            dict(action="GetReachablePositions")
        ).metadata["actionReturn"]
        reachable_positions = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]
        return reachable_positions

    def update_current_state(self, act_texts: List[str]):
        """
        Update the input dictionary with the current state of the environment
        """
        self.new_all_obs = []  # to store previous observations
        for agent_id in range(self.num_agents):
            agent_name = self.agent_names[agent_id]
            obs, obs_list = self.generate_obs_text(agent_id, prefix="")
            self.input_dict[agent_name + "'s observation"] = obs
            state = self.get_agent_state(agent_id)
            self.input_dict[agent_name + "'s state"] = state
            # obs_list = self.convert_str2list(obs)
            self.all_obs.append(obs_list)
            self.input_dict[agent_name + "'s previous observation"] = self.all_obs[
                agent_id
            ]
            self.input_dict[agent_name + "'s previous action"] = act_texts[agent_id]
        self.all_obs = self.new_all_obs

    def step(self, actions: List):
        """
        Execute the actions for all the agents in the environment
        Return the observation for all the agents
        """
        act_successes, act_texts = [], []
        for agent_id in range(self.num_agents):
            error_type = None
            if "NavigateTo" in actions[agent_id]:
                act_success, error_type = self.navigation_step(
                    actions[agent_id], agent_id
                )
            elif actions[agent_id] in ["Done", "Idle"]:
                act_success = True
            # @ coela - added this to handle SendMessage
            elif "SendMessage" in actions[agent_id]:
                act_success = True
            else:
                # TODO: add previous action check here; check if action was taken in the previous step
                action_dict = self.parse_action(actions[agent_id], agent_id)

                self.event = self.controller.step(action_dict)
                act_success = self.event.events[agent_id].metadata["lastActionSuccess"]
                self.previous_success[agent_id] = act_success
                self.step_num[agent_id] += 1
                # TODO
                self.save_frame(agent_id)

            # if action is successful, make the agent_failure_acts list empty
            if act_success:
                self.agent_failure_acts[self.agent_names[agent_id]] = []
                if "PickupObject" in actions[agent_id]:
                    self.inventory[agent_id] = self.get_agent_object_held(agent_id)
            else:
                self.agent_failure_acts[self.agent_names[agent_id]].append(
                    actions[agent_id]
                )

            # add the action and succes to the action and success history
            self.action_history[self.agent_names[agent_id]].append(actions[agent_id])
            self.action_success_history[self.agent_names[agent_id]].append(act_success)
            self.step_nums_history[self.agent_names[agent_id]].append(
                self.step_num[agent_id]
            )

            # NOTE that self.inventory[agent_id] is updated for PutObject in self.get_act_text
            # this means that the inventory is not updated after the PutObject action
            # is executed after the action is successful, so at this point, the inventory
            # is still the same as it was before the action was executed
            self.checker.perform_metric_check(
                actions[agent_id], act_success, self.inventory[agent_id]
            )

            # if self.verbose:
            #   print(f"Completed Subtasks: ")
            #   print("\n".join(self.checker.subtasks_completed))

            act_text = self.get_act_text(
                actions[agent_id], act_success, agent_id, error_type=error_type
            )
            act_texts.append(act_text)
            act_successes.append(act_success)

        self.update_current_state(act_texts)
        return convert_dict_to_string(self.input_dict), act_successes

    def navigation_step(self, action: str, agent_id: int):
        """
        Execute all the navigation actions for the agent in the environment
        Return the observation for all the agents
        action: "NavigateTo(CounterTop_1)"
        """
        self.get_object_in_view(self.event, agent_id)
        object_id = action.split("(")[1].split(")")[0]
        error_type = None
        try:
            actions = self.get_navigation_actions(agent_id, object_id)
        except (KeyError, ValueError) as k:
            # assuming valueerror is from trying to find number in array (if object x_1 exists but not x_77)
            return False, "key"

        if actions is None:
            # nonetype for actions means that object doesn't exist for the agent (not visible, so not navigatable)
            return False, "no-actions"

        act_successes = []
        # @bug - changed to enumerate (to know if we're the first action or not)
        for i, action in enumerate(actions):
            action_dict = self.parse_action(action, agent_id)
            # TODO save frames here
            self.event = self.controller.step(action_dict)
            act_success = self.event.events[agent_id].metadata["lastActionSuccess"]
            self.step_num[agent_id] += 1
            # TODO
            self.save_frame(agent_id)

            # NOTE: Sometimes the LookDown/LookUp action is not successful
            # because we can only move it by 30 degrees in get_shortest_path,
            # but the agent is still able to see the object
            if "Look" in action:
                # print("Look action sucess: ", act_success)
                act_success = True

            # break the loop if the action is not successful since actions are temporally dependent
            # @bug - don't break loop if first action (teleport) is failure (it just means it's at correct position)
            if not act_success and i != 0:
                break
            # @bug - make sure that failures in alignment aren't counted as actions failures (they aren't, just means the agent is already aligned)
            if not "align" in action.lower():
                act_successes.append(act_success)

        # effective failure if not close enough to object (can't interact)
        object_id = self.convert_readable_object_to_id(object_id)
        # this should be restricted to interactable (not it's interactable objects once we act to account for navigateto non-interactable)
        obj_interactable = object_id in self.get_object_in_view(
            self.controller.last_event, agent_id
        )
        # interactability error
        error_type = None if obj_interactable else "non-interactable"

        # print(actions, act_successes)
        # return true only if all the navigation actions are successful
        return all(act_successes) and obj_interactable, error_type

    # Util functions for overhead images & frame capture
    def set_overhead(self, o):
        self.overhead = o

    # get overhead image
    def _get_ceiling_image(self):
        # puts the camera in the ceiling, then puts it back with the robot
        event = self.controller.step("ToggleMapView")
        # toggle back for later
        self.controller.step("ToggleMapView")
        return event.cv2img

    def _write_image(self, pth, img):
        cv2.imwrite(str(pth), img)

    def _create_path(self, pth):
        if not os.path.exists(pth):
            os.makedirs(pth)

    def save_frame(self, agent_id: int):
        agent_dir_pov = self.render_image_path / self.agent_names[agent_id] / "pov"
        agent_dir_overhead = (
            self.render_image_path / self.agent_names[agent_id] / "overhead"
        )
        self._create_path(agent_dir_pov)
        self._create_path(agent_dir_overhead)

        # add overhead view for gif
        if self.overhead:
            img = self._get_ceiling_image()
            pth = agent_dir_overhead / f"frame_{self.step_num[agent_id]}.png"
            self._write_image(pth, img)

        img = self.event.events[agent_id].cv2img
        pth = agent_dir_pov / f"frame_{self.step_num[agent_id]}.png"
        self._write_image(pth, img)

    def get_frame(self, agent_id: int):
        image_path = (
            self.render_image_path
            / self.agent_names[agent_id]
            / "pov"
            / f"frame_{self.step_num[agent_id]}.png"
        )
        return image_path

    def generate_obs_text(
        self, agent_id: int, prefix="I see: "
    ) -> Tuple[str, List[str]]:
        """
        Returns a string that describes the agent's observation
        """
        obs_text = ""
        obs_text += prefix
        objects_in_view = self.get_object_in_view(self.event, agent_id)
        # ['Cabinet_1', 'UpperCabinets_1', 'Egg_1', 'CounterTop_1']
        obs_list = self.get_readable_object_list(objects_in_view)
        self.pre_sum_obs_list[self.agent_names[agent_id]] = obs_list
        obs_text += str(obs_list)
        return obs_text, obs_list

    def get_task_success(self):
        """
        Returns True if the task is done, else False
        """
        pass

    def get_object_in_view(self, event, agent_id: int):
        """
        Get the objects in the agent's view
        This is used to update the object_dict
        This ensures that the agents are not able to interact with objects
        that they haven't encountered in the environment before.
        """
        # obtain all interactable objects (including those out of view)
        interactable_objects = get_interactable_objects(
            self.event.events[agent_id].metadata["objects"]
        )

        # obtain objects the agent currently sees (in view)
        detections = event.events[agent_id].instance_detections2D
        # detections = event.instance_detections2D
        objects_inview = list(detections.instance_masks.keys())

        # obtain intersection of these 2 lists
        interactable_inview_objects = list(
            set(objects_inview).intersection(set(interactable_objects))
        )

        # add all the new objects to the object_dict
        for obj in interactable_inview_objects:
            obj_name, obj_id = self.parse_object(obj)
            if obj_name not in self.object_dict.keys():
                self.object_dict[obj_name] = {}
            if obj_id not in self.object_dict[obj_name].keys():
                id_num = len(self.object_dict[obj_name].keys()) + 1
                self.object_dict[obj_name][obj_id] = id_num
        return interactable_inview_objects
