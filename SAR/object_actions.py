import re
import copy
import torch
import functools
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
from itertools import product
from core import Controller
from misc import extract_number, hashable_with_cache

script_dir = Path(__file__).parent.absolute()

INIT_PARAMS=None
def update_params(object_dict : dict):
    global INIT_PARAMS
    INIT_PARAMS=object_dict

def get(pois = None):
    """ Get all names from poi type: reservoirs, deposits, fires, persons, agents """
    li_params=INIT_PARAMS
    assert li_params is not None, f"Cannot get name of objects if INIT_PARAMS is none."

    if isinstance(pois, str): pois=[pois]

    # if None assume all
    if pois is None:
        pois=list(li_params.keys())

    l=[]
    for poi in pois:
        ll=li_params.get(poi, [])
        ll=list(map(lambda a : a.name, ll))
        l.extend(ll)
    return l # list of names for all poi(s) listed in the pois

def cardinal_directions():
    return Controller.MOVABLE_CARDINAL_DIRECTIONS # includes the Center

def supply_types():
    return list(map(lambda v : v.capitalize(), Controller.SUPPLY_TYPES))

"""
Formatting for actions, required action_args:

'NavigateTo' : to_target_id (location),
'Move' : to_target_id (direction), # Up, Down, Left, Right
'Carry' : from_target_id (person),
'DropOff' : from_target_id (person), to_target_id (deposit),
'StoreSupply' : to_target_id (deposit),
'UseSupply' : from_target_id (fire), supply_type (on fire)
'GetSupply' : from_target_id (deposit or reservoir), supply_type (deposit)
'ClearInventory',
'NoOp',
"""

# ----------------------------

@hashable_with_cache
def all_actions_embeddings(object_dict : dict):

    # update our global INIT_PARAMS variable (that is used for all get(.))
    assert object_dict is not None, f"Error, object dict for embedding all actions is none"
    update_params(object_dict)

    # -- get available names ---
    available_object_names=object_dict.pop('available_object_names')
    # filter out fire objects (w/o _Region_{n}) from navigation
    def is_not_fire(s):
        if ('Fire' in s) and ('_Region' not in s):
            return False
        return True
    available_object_names=list(filter(is_not_fire, available_object_names))

    # --- navigation / movement ---
    # you can navigate to any names, no restriction
    navigate=[f"NavigateTo({o})" for o in available_object_names]
    move=[f"Move({d})" for d in cardinal_directions()]
    explore=["Explore()"]

    # --- carry / drop ---
    carry=[f"Carry({o})" for o in get('persons')]
    dropoff=[f"DropOff({d}, {p})" for p,d in product(get('persons'), get('deposits'))]

    # --- supply ---
    store=[f"StoreSupply({o})" for o in get('deposits')]
    # all fire x supply permutations
    use=[f"UseSupply({f}, {s})" for f,s in product(get('fires'), supply_types())]
    # (deposit x supply_type) + (reservoir)
    getsupply=[f"GetSupply({d}, {s})" for d,s in product(get('deposits'), supply_types())]
    getsupply+=[f"GetSupply({r})" for r in get('reservoirs')]
    # clear
    clear=[f"ClearInventory()"]

    # -- noop(s) --
    # done, idle
    done=[f"Done"]
    idle=[f"Idle"]

    # -----------------------------

    all_actions=[]
    all_actions.extend(navigate)
    all_actions.extend(move)
    all_actions.extend(explore)
    all_actions.extend(carry)
    all_actions.extend(dropoff)
    all_actions.extend(store)
    all_actions.extend(use)
    all_actions.extend(getsupply)
    all_actions.extend(clear)
    all_actions.extend(done)
    all_actions.extend(idle)

    # ----------------------------

    # NOTE: Do NOT use finetuned transformer as it has only been finetuned to work for AI2Thor
    # model = SentenceTransformer(str(Path(__file__).parent.absolute() / "sentence_transformer/finetuned_model"))
    embeddings = torch.FloatTensor(model.encode(all_actions))
    return all_actions, embeddings


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------

def get_closest_feasible_action(action: str, object_dict : dict):
    """To convert actions like 'use sand supply on great fire' to 'UseSupply(GreatFire, Sand)' """
    # NOTE: now get_closest_feasible_action requires object_dict,
    #       but we now don't have to use get_closest_object_id

    # get embeddings - this is cached
    all_actions, embeddings = all_actions_embeddings(copy.deepcopy(object_dict))

    action_embedding = torch.FloatTensor(model.encode([action]))
    scores = torch.cosine_similarity(embeddings, action_embedding)
    max_score, max_idx = torch.max(scores, 0)
    pred_action=all_actions[max_idx]

    # number surgery
    intended_number=extract_number(action)
    actual_number=extract_number(pred_action)

    # Default region case: If number on action non-existant, but it's in predicted, then make default 1
    if (intended_number is None) and (actual_number is not None):
        default_number=str(1)
        pred_action=pred_action.replace(str(actual_number), default_number)
    
    return pred_action
