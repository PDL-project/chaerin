import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re

# Construct the absolute path to the CSV file to address relative imports
script_dir = Path(__file__).parent.absolute()
tsv_path = script_dir / "objects_actions.tsv"
df = pd.read_csv(tsv_path, sep="\t")


def get_openable_objects():
    return df[df["Openable"] == "yes"]["Object Type"].tolist()


def get_pickable_objects():
    return df[df["Pickupable"] == "yes"]["Object Type"].tolist()


def get_toggleable_objects():
    return df[df["On/Off"] == "yes"]["Object Type"].tolist()


def get_receptacle_objects():
    return df[df["Receptacle"] == "yes"]["Object Type"].tolist()


def get_fillable_objects():
    return df[df["Fillable"] == "yes"]["Object Type"].tolist()


def get_sliceable_objects():
    return df[df["Sliceable"] == "yes"]["Object Type"].tolist()


def get_cookable_objects():
    return df[df["Cookable"] == "yes"]["Object Type"].tolist()


def get_breakable_objects():
    return df[df["Breakable"] == "yes"]["Object Type"].tolist()


def get_dirty_objects():
    return df[df["Dirty"] == "yes"]["Object Type"].tolist()


def get_usable_objects():
    return df[df["UsedUp"] == "yes"]["Object Type"].tolist()


all_actions = []
open = [f"OpenObject({obj})" for obj in get_openable_objects()]
close = [f"CloseObject({obj})" for obj in get_openable_objects()]
pick = [f"PickupObject({obj})" for obj in get_pickable_objects()]
put = [f"PutObject({obj})" for obj in get_receptacle_objects()]
toggle_on = [f"ToggleObjectOn({obj})" for obj in get_toggleable_objects()]
toggle_off = [f"ToggleObjectOff({obj})" for obj in get_toggleable_objects()]
# fill = [f"FillObject({obj})" for obj in get_fillable_objects()]
slice = [f"SliceObject({obj})" for obj in get_sliceable_objects()]
clean = [f"CleanObject({obj})" for obj in get_dirty_objects()]
# cook = [f"CookObject({obj})" for obj in get_cookable_objects()]
navigate = [f"NavigateTo({obj})" for obj in df["Object Type"].tolist()]
rotate = [f"Rotate({obj})" for obj in ["Left", "Right"]]
lookup = [f"LookUp({obj})" for obj in [30, 60, 90, 120, 150, 180]]
lookdown = [f"LookDown({obj})" for obj in [30, 60, 90, 120, 150, 180]]
move = [f"Move({obj})" for obj in ["Ahead", "Back", "Left", "Right"]]
done = ["Done"]
idle = ["Idle"]
explore = ["Explore"]

all_actions.extend(pick)
all_actions.extend(put)
all_actions.extend(open)
all_actions.extend(close)
all_actions.extend(toggle_on)
all_actions.extend(toggle_off)
all_actions.extend(slice)
all_actions.extend(clean)
all_actions.extend(navigate)
all_actions.extend(rotate)
all_actions.extend(lookup)
all_actions.extend(lookdown)
all_actions.extend(move)
all_actions.extend(done)
all_actions.extend(idle)
all_actions.extend(explore)


# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer(str(Path(__file__).parent.absolute() / "sentence_transformer/finetuned_model"))
embeddings = torch.FloatTensor(model.encode(all_actions))


def get_closest_feasible_action(action: str):
    """To convert actions like RotateLeft to Rotate(Left)"""
    action_embedding = torch.FloatTensor(model.encode([action]))
    scores = torch.cosine_similarity(embeddings, action_embedding)
    max_score, max_idx = torch.max(scores, 0)
    return all_actions[max_idx]


def get_closest_object_id(action: str, object_dict: dict, preaction : str = None):
    """To convert actions like NavigateTo(Refrigerator) to NavigateTo(Fridge_1)"""
    pred_object_name = action.split("(")[1].split(")")[0]
    pred_object_embedding = torch.FloatTensor(model.encode([pred_object_name]))
    object_list = get_object_list(object_dict)
    object_embeddings = torch.FloatTensor(model.encode(object_list))
    scores = torch.cosine_similarity(object_embeddings, pred_object_embedding)
    max_score, max_idx = torch.max(scores, 0)
    closest_object = object_list[max_idx]
    closest_action = action.replace(pred_object_name, closest_object)

    # -------------------------
    # ADDITION: correction of the object number (if necessary) at the end
    # we have to use the preaction since the action given is from get_closest_feasible_action
    # and doesn't contain any of the number information
    if preaction is not None:

        clamp=lambda v, mi, ma: max(min(v, ma), mi)
        extract_number=lambda s: list(map(int, re.findall(r'\d+',s)))

        intended_numbers=extract_number(preaction)
        actual_numbers=extract_number(closest_action)

        # only if there are intended numbers AND the mapped closest also has a number
        # furthermore, restrict to when there is only 1 number, otherwise behavior undefined
        if len(intended_numbers)==1 and len(actual_numbers)==1:
            # if this isn't a valid object (number is too high? or out of bounds?), then clamp.
            # get all the other objects of that name (all other numbers), and desired between min and max
            closest_object_innumerated=closest_object.split('_')[0]
            closest_object_numbers=[extract_number(o)[0] for o in object_list if (closest_object_innumerated in o)]
            min_nmb,max_nmb=min(closest_object_numbers),max(closest_object_numbers)

            a_nmb=int(actual_numbers[0])
            i_nmb=int(intended_numbers[0])
            # clamp to existing values
            i_nmb=clamp(i_nmb, min_nmb, max_nmb)
            # replace actual w/ intended
            closest_action=closest_action.replace(str(a_nmb), str(i_nmb))

    return closest_action

def get_object_list(object_dict: dict):
    """
    {
        'Ceiling': {'|0|2.6|0': 1},
        'StandardCounterHeightWidth': {'|0|0|-1.48': 1},
        'Floor': {'|+00.00|+00.00|+00.00': 1},
    }
    """
    object_list = []
    for key in object_dict.keys():
        object_name = key
        for num in object_dict[key].values():
            object_list.append(object_name + f"_{num}")
    return object_list


def denumerate_object(object: str):
    """
    Remove object number from object_id
    Example: Bread_1 -> Bread
    """
    if "_" in object:
        object_name, object_id = object.split("_")
    else:
        object_name = object
    return object_name


def split_action(action, two_objs=False):
    """
    Split action string into action and object
    Example: PickupObject(Bread_1) -> PickupObject, Bread_1
    if two_objs=True, then split action string into action and two objects
    NavigateTo(Fridge_1, Bread_1) -> NavigateTo, Fridge_1, Bread_1
    """
    if two_objs:
        act = action.split("(")[0]
        obj1, obj2 = action.split("(")[1].split(")")[0].split(", ")
        return act, obj1, obj2
    else:
        act = action.split("(")[0]
        object = action.split("(")[1].split(")")[0]
        return act, object


def get_all_available_actions(objects, inventory_objects, text=False):
    """
    Get a list of all feasible actions at the current state based on
    all the objects visible and in the inventory
    text: bool
        if True, then return actions in text format
        else return in function format
    NOTE: This is used in CoELA"""
    if text:
        feasible_actions = [
            "rotate left",
            "rotate right",
            "move ahead",
            "move backward",
            "move left",
            "move right",
        ]
    else:
        feasible_actions = [
            "Rotate(Left)",
            "Rotate(Right)",
            "Move(Ahead)",
            "Move(Back)",
            "Move(Left)",
            "Move(Right)",
        ]
    for obj in objects:
        denumerated_obj = denumerate_object(obj)
        if text:
            feasible_actions.append(f"navigate to {obj}")
        else:
            feasible_actions.append(f"NavigateTo({obj})")
        # openable and closeable objects
        if denumerated_obj in get_openable_objects():
            if text:
                feasible_actions.append(f"open {obj}")
                feasible_actions.append(f"close {obj}")
            else:
                feasible_actions.append(f"OpenObject({obj})")
                feasible_actions.append(f"CloseObject({obj})")
        # pickable objects
        if denumerated_obj in get_pickable_objects():
            if text:
                feasible_actions.append(f"pick up {obj}")
            else:
                feasible_actions.append(f"PickupObject({obj})")
        # receptacle objects
        if denumerated_obj in get_receptacle_objects() and len(inventory_objects) > 0:
            if text:
                feasible_actions.append(
                    f"put {(', ').join(inventory_objects)} on/in {obj}"
                )
            else:
                feasible_actions.append(f"PutObject({obj})")
        # toggleable objects
        if denumerated_obj in get_toggleable_objects():
            if text:
                feasible_actions.append(f"turn on {obj}")
                feasible_actions.append(f"turn off {obj}")
            else:
                feasible_actions.append(f"ToggleObjectOn({obj})")
                feasible_actions.append(f"ToggleObjectOff({obj})")
        # sliceable objects
        if denumerated_obj in get_sliceable_objects():
            if text:
                feasible_actions.append(f"slice {obj}")
            else:
                feasible_actions.append(f"SliceObject({obj})")
        # cleanable objects
        if denumerated_obj in get_dirty_objects():
            if text:
                feasible_actions.append(f"clean {obj}")
            else:
                feasible_actions.append(f"CleanObject({obj})")
    return feasible_actions


def inverse_act2text(action: str):
    """
    Convert action to text
    Example: PickupObject(Bread_1) -> pick up Bread_1
    Rotate(Right) -> rotate right
    """
    act, object = split_action(action, two_objs=False)
    if "PickupObject" in action:
        return f"pick up {object}"
    elif "PutObject" in action:
        return f"put {object}"
    elif "OpenObject" in action:
        return f"open {object}"
    elif "CloseObject" in action:
        return f"close {object}"
    elif "ToggleObjectOn" in action:
        return f"turn on {object}"
    elif "ToggleObjectOff" in action:
        return f"turn off {object}"
    elif "SliceObject" in action:
        return f"slice {object}"
    elif "CleanObject" in action:
        return f"clean {object}"
    elif "NavigateTo" in action:
        return f"navigate to {object}"
    elif "Rotate" in action:
        return f"rotate {object.lower()}"
    elif "LookUp" in action:
        return f"look up {object.lower()} degrees"
    elif "LookDown" in action:
        return f"look down {object.lower()} degrees"
    elif "Move" in action:
        return f"move {object.lower()}"
    elif "Done" in action:
        return "done"
    elif "Idle" in action:
        return "idle"
    elif "SendMessage" in action:
        message = action.split("(")[1].split(")")[0]
        return f"send message: {message}"


def inverse_subtask2text(action: str, two_objs: bool):
    """
    Convert subtask actions to text in past tense
    NOTE: Used in CoELA baseline
    Example: PickupObject(Bread_1) -> picked up Bread_1
    NavigateTo(Fridge_1, Bread_1) -> navigated to Fridge_1 with Bread_1 in hand
    Rotate(Right) -> rotated right
    two_objs: bool, if True, then the action has two objects
    Example: PutObject(Fridge_1, Bread_1) -> put Bread_1 in Fridge_1
    """
    if two_objs:
        act, obj1, obj2 = split_action(action, two_objs=True)
        if "PutObject" in action:
            return f"put {obj2} in {obj1}"
        elif "NavigateTo" in action:
            return f"navigated to {obj1} with {obj2} in hand"
    else:
        act, object = split_action(action, two_objs=False)
        if "PickupObject" in action:
            return f"picked up {object}"
        elif "PutObject" in action:
            return f"put {object}"
        elif "OpenObject" in action:
            return f"opened {object}"
        elif "CloseObject" in action:
            return f"closed {object}"
        elif "ToggleObjectOn" in action:
            return f"turned on {object}"
        elif "ToggleObjectOff" in action:
            return f"turned off {object}"
        elif "SliceObject" in action:
            return f"sliced {object}"
        elif "CleanObject" in action:
            return f"cleaned {object}"
        elif "NavigateTo" in action:
            return f"navigated to {object}"
        else:
            return action


def get_interactable_objects(object_metadata: list):
    """
    Get list of interactable objects in environment
    """
    interactable_objs = []
    for obj_dict in object_metadata:
        interactable_keys = [
            key for key in obj_dict.keys() if "able" in key and obj_dict[key] == True
        ]
        if interactable_keys:
            interactable_objs.append(obj_dict["objectId"])

    return interactable_objs
