import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

from thortils.navigation import get_shortest_path_to_object

# @bug - add import to measure needed rotation
from thortils.utils import closest, closest_angles
from AI2Thor.object_actions import get_interactable_objects


def convert_dict_to_string_with_double_braces(input_dict: Dict) -> str:
    # needed to work with langchain formatting

    string_representation = str(input_dict)
    string_with_double_braces = string_representation.replace("{", "{{").replace(
        "}", "}}"
    )

    return string_with_double_braces


def convert_to_string_with_double_braces(input_string: str) -> str:
    # needed for langchain formatting
    string_with_double_braces = input_string.replace("{", "{{").replace("}", "}}")
    return string_with_double_braces


def convert_dict_to_string(input_dict) -> str:
    """
    add new lines for each key value pair
    Example output:
    {Task: bring a tomato, lettuce and bread to the countertop to make a sandwich
    Alice's observation: I see: ['Cabinet_1', 'UpperCabinets_1', 'StandardCounterHeightWidth_1', 'Wall_1', 'Ceiling_1', 'Fridge_1', 'Toaster_1', 'CoffeeMachine_1', 'CounterTop_1']
    Alice's state: I am at co-ordinates: (-1.00, 0.90, 1.00) and I am holding nothing
    Bob's observation: I see: ['Cabinet_1', 'Sink_1', 'Mug_1', 'UpperCabinets_1', 'StandardCounterHeightWidth_1', 'Wall_1', 'Window_1', 'Fridge_1', 'Toaster_1', 'CoffeeMachine_1', 'PaperTowelRoll_1', 'CounterTop_1']
    Bob's state: I am at co-ordinates: (-1.00, 0.90, 0.00) and I am holding nothing
    History: {'Observations': {'Alice': [], 'Bob': []}, 'Actions': {'Alice': [], 'Bob': []}, 'Action Success': {'Alice': [], 'Bob': []}} }
    """
    return "{" + "\n".join(f"{k}: {v}, " for k, v in input_dict.items()) + "}"


class BaseEnv:
    def __init__(self):
        pass

    def random_spawn(self, seed: int = 0):
        """
        https://ai2thor.allenai.org/ithor/documentation/objects/domain-randomization/#initial-random-spawn
        """
        self.controller.step(action="InitialRandomSpawn", randomSeed=seed)

    def agent_init_pos(self, agent_id: int):
        self.event = self.controller.step(
            dict(
                action="Teleport",
                position=dict(x=1.5, y=0.9, z=-1.5),
                rotation=dict(x=0, y=270, z=0),
                agentId=int(agent_id),
            )
        )

    def check_cache(self, path: str):
        if os.path.exists(path):
            print("Loading LLM cache")
            cache = pickle.load(open(self.path, "rb"))
        else:
            # print("LLM cache not found, creating new")
            cache = {}
        return cache

    def create_save_dirs(self):
        # self.action_llm_cache = self.check_cache(self.action_llm_cache_path)
        self.obs_summary_llm_cache = self.check_cache(self.obs_summary_llm_cache_path)
        # self.render_image_path = args.render_image_path
        self.render_image_path = (
            Path("render_images")
            / self.task
            / self.args.scene_name
            / str(self.args.num_agents)
        )
        self.action_dir_path = (
            Path("actions")
            / self.task
            / self.args.scene_name
            / str(self.args.num_agents)
        )

        if not os.path.exists(self.render_image_path):
            os.makedirs(self.render_image_path)
        if not os.path.exists(self.action_dir_path):
            os.makedirs(self.action_dir_path)

    def get_agent_state(self, agent_id: int):
        """
        Return a string describing the agent's state
        """
        agent_state = f"I am at co-ordinates: {self.get_agent_position(agent_id)} facing {self.get_agent_rotation(agent_id)} and I am holding {self.inventory[agent_id]}"
        return agent_state

    def get_agent_position(self, agent_id: int) -> str:
        """
        Return a string describing the agent's position
        """
        pos_dict = self.event.events[agent_id].metadata["agent"]["position"]
        position = f"({pos_dict['x']}, {pos_dict['z']})"  # y is height
        # position = f"({pos_dict['x']:.2f}, {pos_dict['y']:.2f}, {pos_dict['z']:.2f})"
        return position

    def get_agent_rotation(self, agent_id: int) -> str:
        """
        Return a string describing the agent's rotation
        """
        rot = self.event.events[agent_id].metadata["agent"]["rotation"]["y"]
        # in some scenarios, the rotation is not exactly 0, 90, 180, 270
        # so we need to round it to the nearest integer
        rot = int(np.round(rot))
        # @bug 0 give approximation of direction (w/ closest_angles function)
        rot = closest_angles([0, 90, 180, 270], rot)

        if rot == 0:
            rotation = "north"
        elif rot == 90:
            rotation = "east"
        elif rot == 180:
            rotation = "south"
        elif rot == 270:
            rotation = "west"

        return rotation

    def get_agent_object_held(self, agent_id: int):
        """
        Return a string describing the agent's object held
        """
        held_obj = self.event.events[agent_id].metadata["inventoryObjects"]
        assert len(held_obj) <= 1, "Agent can only hold one object at a time"
        if len(held_obj) == 0:
            return "nothing"
        else:
            obj_dict = held_obj[
                0
            ]  # {'objectId': 'Apple|-00.47|+01.15|+00.48', 'objectType': 'Apple'}
            obj_name, obj_id = self.parse_object(obj_dict["objectId"])
            obj_num = self.object_dict[obj_name][obj_id]
            return obj_name + "_" + str(obj_num)

    def convert_str2list(self, obs_str: str):
        """
        Convert a string of the form "['Cabinet_1', 'CounterTop_1']" to a list of the form ['Cabinet_1', 'CounterTop_1']
        """
        obs_list = (
            obs_str.replace("[", "").replace("]", "").replace("'", "").split(", ")
        )
        return obs_list

    def parse_action(self, action: str, agent_id: int):
        """
        Convert
        • 'Move(Ahead)' to 'MoveAhead'
        • 'Rotate(Left)' to 'RotateLeft'
        • 'PickupObject(object_id)' to 'PickupObject', 'object_id'
        • 'PutObject(receptacle_id)' to 'PutObject', 'receptacle_id'
        • 'OpenObject(object_id)' to 'OpenObject', 'object_id'
        • 'CloseObject(object_id)' to 'CloseObject', 'object_id'
        • 'SliceObject(object_id)' to 'SliceObject', 'object_id'
        • 'ToggleObjectOn(object_id)' to 'ToggleObjectOn', 'object_id'
        • 'ToggleObjectOff(object_id)' to 'ToggleObjectOff', 'object_id'
        • 'LookUp(angle)' to 'LookUp', 'angle'
        • 'LookDown(angle)' to 'LookDown', 'angle'
        • 'CleanObject(object_id)' to 'CleanObject', 'object_id'
        • 'Explore() to explore
        # @bug - added 'AlignOrientation(pitch,yaw,z)' to 'Teleport' rotation dict
        """

        action_dict = {}
        if action in ["Explore"]:
            action_dict["action"] = action
            action_dict["agentId"] = agent_id
            return action_dict

        action_name = action.split("(")[0]
        action_inner = action.split("(")[1].replace(")", "")

        if action_name in ["Move", "Rotate"]:
            direction = action.split("(")[1].split(")")[0]
            action = action_name + direction
            if action_name == "Move":
                assert action in self.move_actions, "Invalid action"
            elif action_name == "Rotate":
                assert action in self.rotate_actions, "Invalid action"
            action_dict["action"] = action
            action_dict["agentId"] = agent_id
        elif action_name in [
            "PickupObject",
            "PutObject",
            "OpenObject",
            "CloseObject",
            "SliceObject",
            "ToggleObjectOn",
            "ToggleObjectOff",
            "CleanObject",
        ]:
            # here object_id will be in the form of "Cabinet_1"
            # need to convert back to "Cabinet|-01.85|+02.02|+00.38"

            object_id = action.split("(")[1].split(")")[0]

            object_id = self.convert_readable_object_to_id(object_id)
            action_dict["action"] = action_name
            action_dict["agentId"] = agent_id
            action_dict["objectId"] = object_id

            # @tt - special forceaction for putobject in fridge (avoid AI2Thor issue #1210 - https://github.com/allenai/ai2thor/issues/1210)
            if action_name == "PutObject" and "fridge" in action_inner.lower():
                action_dict["forceAction"] = True

        elif action_name in ["LookUp", "LookDown"]:
            angle = action.split("(")[1].split(")")[0]
            action_dict["action"] = action_name
            action_dict["agentId"] = agent_id
            action_dict["degrees"] = angle
        # @bug - parsing the alignorientation action
        elif action_name in ["AlignOrientation"]:
            # AlignOrientation(pitch,yaw) -> do this for final orientation alignment to face object
            # full teleportation, change camera horizon depending on variable z
            action_dict["agentId"] = agent_id
            # TODO: does adding current pitch fix it?
            pitch, yaw, z = action.split("(")[1].split(")")[0].split(",")
            agent_metadata = self.event.events[agent_id].metadata["agent"]
            horizon = agent_metadata["cameraHorizon"]

            action_dict["action"] = "TeleportFull"
            action_dict["rotation"] = dict(x=pitch, y=yaw, z=0)
            action_dict["position"] = agent_metadata["position"]
            z = eval(z)  # str -> bool
            if z:
                action_dict["horizon"] = 0
            else:
                action_dict["horizon"] = horizon

            action_dict["standing"] = agent_metadata["isStanding"]
            action_dict["forceAction"] = True

            # self.input_dict["open subtasks"]

        return action_dict

    def parse_object(self, object_str: str):
        """Example: Cabinet|+01.08|+00.90|-00.77
        Return 'Cabinet', '+01.08|+00.90|-00.77'
        """
        object_name = object_str.split("|")[0]
        object_str_id = object_str.replace(object_name, "")
        return object_name, object_str_id

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

    def get_readable_object_list(self, object_list: List[str]):
        """
        Convert
        ['Cabinet|-01.85|+02.02|+00.38', 'UpperCabinets|0|0|0',
        'StandardCounterHeightWidth|-1.88|0|0.08', 'Wall|2.38|0|-0.09999999',
        'Ceiling|0|0|0', 'Fridge|-02.10|+00.00|+01.07',
        'Toaster|-01.84|+00.90|+00.13', 'CoffeeMachine|-01.98|+00.90|-00.19',
        'CounterTop|-01.87|+00.95|-01.21']
        to
        ['Cabinet_1', 'UpperCabinets_1', 'StandardCounterHeightWidth_1', 'Wall_1',
        'Ceiling_1', 'Fridge_1', 'Toaster_1', 'CoffeeMachine_1', 'CounterTop_1']
        """
        readable_object_list = []
        for obj in object_list:
            obj_name, obj_id = self.parse_object(obj)
            obj_num = self.object_dict[obj_name][obj_id]
            readable_object_list.append(obj_name + "_" + str(obj_num))
        return readable_object_list

    def convert_readable_object_to_id(self, object_name: str):
        """
        object_name = 'Cabinet_1'
        returns 'Cabinet|-01.85|+02.02|+00.38'
        """
        obj_name, obj_num = object_name.split("_")
        obj_dict = self.object_dict[obj_name]
        obj_id = list(obj_dict.keys())[list(obj_dict.values()).index(int(obj_num))]
        return obj_name + obj_id

    def convert_object_id_to_readable(self, object_id: str):
        """
        object_id = 'Cabinet|-01.85|+02.02|+00.38'
        returns 'Cabinet_1'
        """
        obj_name, obj_id = self.parse_object(object_id)
        obj_num = self.object_dict[obj_name][obj_id]
        return obj_name + "_" + str(obj_num)

    def get_act_text(
        self, action: str, act_success: bool, agent_id: int, error_type: str = None
    ) -> str:
        """Get a text describing what changes the action taken in the previous step"""
        action_name = action.split("(")[0]

        # mobility actions
        if action_name in ["Move", "Rotate"]:
            direction = action.split("(")[1].split(")")[0]
            if act_success:
                if action_name == "Move":
                    act_text = f"I moved {direction.lower()} in the previous step and was successful."
                elif action_name == "Rotate":
                    act_text = f"I rotated {direction.lower()} in the previous step and was successful."
            else:
                if action_name == "Move":
                    act_text = f"I tried to move {direction.lower()} in the previous step but was unsuccessful."
                elif action_name == "Rotate":
                    act_text = f"I tried to rotate {direction.lower()} in the previous step but was unsuccessful."

        # done and idle actions
        if action_name in ["Done", "Idle"]:
            if action_name == "Done":
                act_text = (
                    "I completed the task in the previous step and was successful."
                )
            elif action_name == "Idle":
                act_text = "I was idle in the previous step."

        # camera angle actions
        if action_name in ["LookUp", "LookDown"]:
            angle = action.split("(")[1].split(")")[0]
            if act_success:
                if action_name == "LookUp":
                    act_text = f"I looked up {angle} degrees in the previous step and was successful."
                elif action_name == "LookDown":
                    act_text = f"I looked down {angle} degrees in the previous step and was successful."
            else:
                if action_name == "LookUp":
                    act_text = f"I tried to look up {angle} degrees in the previous step but was unsuccessful."
                elif action_name == "LookDown":
                    act_text = f"I tried to look down {angle} degrees in the previous step but was unsuccessful."

        # navigation action
        if action_name == "NavigateTo":
            object_id = action.split("(")[1].split(")")[0]
            # TODO: give more feedback for type of failure
            if act_success:
                act_text = f"I navigated to {object_id} in the previous step and was successful."
            elif error_type == "key":
                act_text = f"I tried to navigate to {object_id} in the previous step but {object_id} doesn't exist, unsuccessful."
            elif error_type == "non-interactable":
                act_text = f"I tried to navigate to {object_id} in the previous step but {object_id} due to blockage object won't be interactable, unsuccessful."
            elif error_type == "no-actions":
                act_text = f"I tried to navigate to {object_id} in the previous step but {object_id} object but there isn't a clear path to it, unsuccessful."
            else:
                act_text = f"I tried to navigate to {object_id} in the previous step but was unsuccessful."

        # object interaction actions
        if action_name in [
            "PickupObject",
            "PutObject",
            "OpenObject",
            "CloseObject",
            "SliceObject",
            "ToggleObjectOn",
            "ToggleObjectOff",
            "CleanObject",
        ]:
            object_id = action.split("(")[1].split(")")[0]
            if act_success:
                if action_name == "PutObject":
                    act_text = f"I put the {self.inventory[agent_id]} in my hand on {object_id} in the previous step and was successful"
                    # if we are using action_failure then we need inventory object so update it then
                    # if not self.use_action_failure:
                    # once we update the act_text, we need to update the inventory
                    self.inventory[agent_id] = "nothing"
                elif action_name == "CloseObject":
                    act_text = (
                        f"I closed {object_id} in the previous step and was successful"
                    )
                elif action_name == "OpenObject":
                    act_text = (
                        f"I opened {object_id} in the previous step and was successful"
                    )
                elif action_name == "PickupObject":
                    act_text = f"I picked up {object_id} in the previous step and was successful"
                elif action_name == "SliceObject":
                    act_text = (
                        f"I sliced {object_id} in the previous step and was successful"
                    )
                elif action_name == "ToggleObjectOn":
                    act_text = f"I turned on {object_id} in the previous step and was successful"
                elif action_name == "ToggleObjectOff":
                    act_text = f"I turned off {object_id} in the previous step and was successful"
                elif action_name == "CleanObject":
                    act_text = (
                        f"I cleaned {object_id} in the previous step and was successful"
                    )
            else:
                if action_name == "PutObject":
                    act_text = f"I tried to put the {self.inventory[agent_id]} in my hand on {object_id} in the previous step but was unsuccessful"
                elif action_name == "CloseObject":
                    act_text = f"I tried to close {object_id} in the previous step but was unsuccessful"
                elif action_name == "OpenObject":
                    act_text = f"I tried to open {object_id} in the previous step but was unsuccessful"
                elif action_name == "PickupObject":
                    act_text = f"I tried to pick up {object_id} in the previous step but was unsuccessful"
                elif action_name == "SliceObject":
                    act_text = f"I tried to slice {object_id} in the previous step but was unsuccessful"
                elif action_name == "ToggleObjectOn":
                    act_text = f"I tried to turn on {object_id} in the previous step but was unsuccessful"
                elif action_name == "ToggleObjectOff":
                    act_text = f"I tried to turn off {object_id} in the previous step but was unsuccessful"
                elif action_name == "CleanObject":
                    act_text = f"I tried to clean {object_id} in the previous step but was unsuccessful"

        # communication actions
        if action_name in [
            "SendMessage",
        ]:
            message = action.split("(")[1].split(")")[0]
            if act_success:
                act_text = f"I sent a message {message} in the previous step and was successful."
            else:
                act_text = f"I tried to send a message {message} in the previous step but was unsuccessful."

        return act_text

    def get_act_failure_text(self, actions: List[str], agent_id: int) -> str:
        """Get a text describing that the previous actions failed
        actions: List of failed actions, agent_id: int
        actions = [<action_1>, <action_2>, ..., <action_n>]
        Example: Previously, I have tried to <action_1>, <action_2>, ..., <action_n> but was unsuccessful.
        Previously, I have tried to put the Egg_1 in my hand on CounterTop_1, move ahead, rotate left but was unsuccessful.
        """
        num_failures = len(actions)
        if num_failures == 0:
            return "None"
        failure_text = "Previously, I have tried to "
        end_phrase = ", but was unsuccessful"
        num_acts_remaining = num_failures
        for action in actions:
            action_name = action.split("(")[0]
            # Move and Rotate actions
            if action_name in ["Move", "Rotate"]:
                direction = action.split("(")[1].split(")")[0]
                if action_name == "Move":
                    act_text = f"move {direction.lower()}"
                elif action_name == "Rotate":
                    act_text = f"rotate {direction.lower()}"
            # navigation actions
            if action_name == "NavigateTo":
                object_id = action.split("(")[1].split(")")[0]
                act_text = f"navigate to {object_id}"
            # LookUp and LookDown actions
            if action_name in ["LookUp", "LookDown"]:
                angle = action.split("(")[1].split(")")[0]
                if action_name == "LookUp":
                    act_text = f"look up {angle} degrees"
                elif action_name == "LookDown":
                    act_text = f"look down {angle} degrees"
            # Pickup, Put, Open, Close, Slice, Toggle actions
            if action_name in [
                "PickupObject",
                "PutObject",
                "OpenObject",
                "CloseObject",
                "SliceObject",
                "ToggleObjectOn",
                "ToggleObjectOff",
                "CleanObject",
            ]:
                object_id = action.split("(")[1].split(")")[0]
                if action_name == "PutObject":
                    act_text = (
                        f"put the {self.inventory[agent_id]} in my hand on {object_id}"
                    )
                    ## once we update the failure_act_text, we need to update the inventory
                    # self.inventory[agent_id] = "nothing"
                elif action_name == "CloseObject":
                    act_text = f"close {object_id}"
                elif action_name == "OpenObject":
                    act_text = f"open {object_id}"
                elif action_name == "PickupObject":
                    act_text = f"pick up {object_id}"
                elif action_name == "SliceObject":
                    act_text = f"slice {object_id}"
                elif action_name == "ToggleObjectOn":
                    act_text = f"turn on {object_id}"
                elif action_name == "ToggleObjectOff":
                    act_text = f"turn off {object_id}"
                elif action_name == "CleanObject":
                    act_text = f"clean {object_id}"
            failure_text += act_text
            # there are more actions remaining
            if num_acts_remaining > 1:
                failure_text += ", "
            # there are only two actions remaining so we need to add an "and" for the last two actions
            if num_acts_remaining == 0 and num_failures > 1:
                failure_text += " and "

            num_acts_remaining -= 1
        failure_text += end_phrase

        return failure_text

    def convert_navigation_acts(self, action: Tuple[str, Tuple]) -> str:
        # action = ('LookUp', (0.0, 0.0, -30))
        """
        Convert the navigator outputs to environment actions
        """
        act = action[0]
        params = action[1]
        if act == "MoveAhead":
            return "Move(Ahead)"
        if act == "MoveRight":
            return "Move(Right)"
        if act == "MoveLeft":
            return "Move(Left)"
        if act == "MoveBack":
            return "Move(Back)"
        if act == "RotateRight":
            return "Rotate(Right)"
        if act == "RotateLeft":
            return "Rotate(Left)"
        if act == "LookUp":
            # action = ('LookUp', (0.0, 0.0, -30))
            angle = params[2]
            if angle <= 0:
                return f"LookDown({np.abs(angle)})"
            else:
                return f"LookUp({np.abs(angle)})"
        if act == "LookDown":
            angle = params[2]
            if angle <= 0:
                return f"LookUp({np.abs(angle)})"
            else:
                return f"LookDown({np.abs(angle)})"

        # @bug - rotate to correct position
        if act == "AlignOrientation":
            pitch, yaw, z = params[0], params[1], params[2]
            return f"AlignOrientation({pitch},{yaw},{z})"

    def get_navigation_actions(
        self, agent_id: int, object_name: str
    ) -> List[Tuple[str, Tuple]]:
        # @bug - wrap agent rotation at the end of this function (face to the direction correctly)
        # @bug - initially approximate the agent direction into one of the four 'cannonical' rotations
        from thortils.constants import H_ANGLES, V_ANGLES

        cur_rot = self.event.events[agent_id].metadata["agent"]["rotation"]
        _yaw_initial = closest_angles(H_ANGLES, cur_rot["y"])
        _pitch_initial = closest_angles(V_ANGLES, cur_rot["x"])
        upd_rotation = dict(x=_pitch_initial, y=_yaw_initial, z=cur_rot["z"])

        object_id = self.convert_readable_object_to_id(object_name)
        cur_pos = self.event.events[agent_id].metadata["agent"]["position"]
        # @tt - camera horizon re-correction
        cameraHorizon = self.event.events[agent_id].metadata["agent"]["cameraHorizon"]
        cameraHorizon = closest_angles(V_ANGLES, cameraHorizon)

        def pitch_act(cameraHorizon, act=True):
            pitch_action = None
            if cameraHorizon > 0:
                pitch_action = f"LookUp({cameraHorizon})"
            elif cameraHorizon < 0:
                pitch_action = f"LookDown({cameraHorizon})"
            # act now - before we try to find path (will not work otherwise)
            if pitch_action is not None and act:
                # TODO: will this be added to the list for the env (bad)
                action_dict = self.parse_action(pitch_action, agent_id)
                self.event = self.controller.step(action_dict)

        # conditionally choose the pitch action - fix navigation issues
        pitch_act(cameraHorizon)

        cur_pos = (cur_pos["x"], cur_pos["y"], cur_pos["z"])
        # cur_rot = self.event.events[agent_id].metadata["agent"]["rotation"]
        # @bug - use the sneaky 'fixed up' one that was made
        cur_rot = upd_rotation
        cur_rot = (cur_rot["x"], cur_rot["y"], cur_rot["z"])

        # @bug - create align intial action
        xx, yy, zz = cur_rot
        # use zz as conditional value (True if change camera horizon, False if not)
        align_initial_action = f"AlignOrientation({xx},{yy},{False})"

        # @muliagent - pass in non-current-agent's position(s)
        other_agents = [
            self.event.events[i].metadata["agent"]["position"]
            for i in range(self.args.num_agents)
            if i != agent_id
        ]
        poses, actions = get_shortest_path_to_object(
            self.controller, other_agents, object_id, cur_pos, cur_rot, return_plan=True
        )
        if actions is None:
            # TODO: rotate back to same lookup/down if it still fails (welp)
            pitch_act(-cameraHorizon)
            return None

        # @bug - getting it from the get_shortest_path_to_object function (at the end)
        goal_pitch, goal_yaw = poses.pop()

        convert_actions = [self.convert_navigation_acts(action) for action in actions]
        # @bug - add wrapper rotations to the beginning & end of actions
        convert_actions.insert(0, align_initial_action)

        # if there are any actions, not None (avoid error)
        if goal_pitch is not None:
            align_final_action = f"AlignOrientation({goal_pitch},{goal_yaw},{False})"
            convert_actions.append(align_final_action)

        return convert_actions
