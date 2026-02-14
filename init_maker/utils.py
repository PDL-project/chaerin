import argparse
import inspect
import importlib
import re
import os, sys
from pathlib import Path

def all_objects(env):
    return env.controller.last_event.metadata["objects"]

def compare(a,b):
    # bigger name, name
    # compare string for objects (true if 'equal')
    a,b=a.lower().strip(), b.lower().strip()
    return b in a

def get_object_from_id(all_objs, objectId):
    for obj in all_objs:
        if obj['objectId'] == objectId:
            return obj
    return None

def get_id(env, name, ignore=False, exclusive=True):
    """
    Gets the id for object with name 'name'
    - name - name of object (e.g. 'tomato')
    - ignore - if True, give a list of all the ids that apply. 
           If False, assert that there is only 1 applicable id and output that one
    - exclusive - if True, be stricter about name (butterknife becomes different than knife)
                    if you want all knives (regardless) set this to False and ignore to True

    e.g. get_id('knife', ignore=True, exclusive=True) -> will return list of 'knife' objects only (NO 'butterknife')
    e.g. get_id('knife', ignore=True, exclusive=False) -> will return list of 'knife' objects only (including 'butterknife')
    """

    ids=[]
    for o in all_objects(env):
        if compare(o['objectId'].split('|')[0], name):
            # symmetric process (remove extraneous things)
            if exclusive and not compare(name, o['objectId'].split('|')[0]):
                continue
            ids.append(o['objectId'])
    assert len(ids)>0, f"object with name {name} not found"
    if ignore==False:
        assert len(ids)==1, f'more than 1 object with name {name} found (get_id): {ids}. If this is ok, change to \'ignore=True\''
        return ids[0]
    return ids

def get_size(env):
    """
    Get the x,y,z size of the current scene
    output is dictionary with keys 'x','y','z'
    """
    return env.event.metadata['sceneBounds']['size']

def get_object(env, obj_name, ignore=False, exclusive=True):
    """
    Get the object metadata for obj with name 'obj_name', NOT with id 'obj_name'
        e.g. get_object('apple')
    See docs above for ignore & exclusive meaning
    """
    objs=[]
    for o in all_objects(env):
        if compare(o['objectId'].split('|')[0], obj_name):
            if exclusive and not compare(obj_name, o['objectId'].split('|')[0]):
                continue
            objs.append(o)
    if ignore==False:
        assert len(objs)==1, f'more than 1 object with name {name} found (get_object): {ids}. If this is ok, change to \'ignore=True\''
        return objs[0] 
    return objs

def get_position(env, obj_name, ignore=False, exclusive=True):
    """
    Get position of object, see docs above for ignore & exclusive meaning
    """
    return get_object(env, obj_name,ignore=ignore, exclusive=exclusive)['position']

def m_offset_position(env, obj_position, off_pos):
    x,y,z=off_pos
    x+=obj_position['x']
    y+=obj_position['y']
    z+=obj_position['z']
    return dict(zip(obj_position.keys(), [x,y,z]))


def offset_position(env, obj_name, pos):
    """
    Given object w/ name obj_name, add the offset (x,y,z) from array 'pos'
    and return the offset-ed position as a dict

    x,y,z correspondance in the plot:
    -x : horizontal motion
    -y : out of plane motion (height of object)
    -z : vertical motion

    When setting position, one can put the height 'y' above a table and it'll fall down (no need to get exact)
    """
    obj_position=get_position(env, obj_name)
    return m_offset_position(env, obj_position, pos)

def abs_offset(env, prop):
    """
    Given proportion array (inputs from [0,1]), 
    get corresponding delta change in absolute distances
    e.g. abs_offset([.1, .1, .1]) -> array of 10% of size of room in x,z,y
    """
    sizes=get_size(env)
    return [v*prop[i] for i,v in enumerate(sizes.values())]

def rel_offset(env, diff):
    sizes=get_size(env)
    return [diff[i]/v for i,v in enumerate(sizes.values())]

# helper function
def fill_array(v):
    """
    Fill x,y,z array with value 'v'
    """
    return [v]*3

def overhead(env):
    """ Get overhead view """
    env.controller.step(action='ToggleMapView')
    
# extract number from string ('FloorPlan201' -> 201)
extract_number=lambda s: re.findall(r'\d+',s)[0]

def write_meta(place_objects=None, actions=None, forceAction=True, evalmode=False):
    """
    This is the main function that creates the python file with the auto-initialized code.
    arguments:
        -place_objects (dictionary (w/ keys) object_id and (w/ values) position dict (output from offset_position(...))
            -e.g. {'Apple|1|1|1' : {'x' : .2334, 'y' : .324, 'z' : -0.0132}}
        -actions (dict (w/ keys) action name and (w/ values) obj id *list* to apply that action)
            -e.g. {'SliceObject': ['Apple|1|1|1', 'Tomato|1|1|1'], 'OpenObject': ['Fridge|1|1|1']}
        -forceAction - boolean value whether to force action or not
        -evalmode - if true, give in format so code can be evaluated (list of function calls) (called from controller, so define that var first)
    """
    
    s=""
    s+="""
    \"""Pre-initialize the environment for the task.

    Args:
        event: env.event object
        controller: ai2thor.controller object

    Returns:
        event: env.event object
    \"""\n
        """

    eval_list = []

    if place_objects is not None:
        for (_id, pos) in (place_objects.items()):
            s+="""
    event=controller.step(
    action=\'PlaceObjectAtPoint\',
    objectId=\'{_id}\',
    position={pos}
    )
    """.format(_id=_id, pos=pos)
            # for eval list
            eval_list.append(
            "controller.step(action=\'PlaceObjectAtPoint\', objectId=\'{_id}\', position={pos})".format(_id=_id, pos=pos)
            )

    if actions is not None:
        for actn, _ids in (actions.items()):
            for _id in _ids:
                s+="""
    event=controller.step(
    action=\'{actn}\',
    objectId=\'{_id}\',
    forceAction={forceAction}
    )
    """.format(actn=actn, _id=_id, forceAction=forceAction)
            # for eval list
            eval_list.append(
            "controller.step(action=\'{actn}\', objectId=\'{_id}\', forceAction={forceAction})".format(actn=actn, _id=_id, forceAction=forceAction)
            )

    s+="""
    return event
    """

    if evalmode:
        return eval_list
    return s


def wrap_meta_result(fs : str, docs : str = "", path : str = None):
    # fs -> output string from write_meta
    # docs -> documentation for the preinit
    # path -> "task_dir/file_name" (task_dir as in json file, file_name for pre-init given floorplan)

    func_header="""
    def preinit(self, event, controller):"""
    func_body=fs.split('\n')
    func=func_header+'\n    '.join(func_body)
    cls="""
class SceneInitializer:
    def __init__(self) -> None:
        pass
        """

    docs=f"""
\"""{docs}\"""
    """
    content=docs+cls+func

    path_prepend='./init_maker_easy/inits/'
    task_dir,fn=path.split('/')
    pt=path_prepend+task_dir+'/'
    if not os.path.exists(pt):
        os.makedirs(pt)

    fn=pt+fn+'.py'
    with open(fn, 'w') as f:
        f.write(content)
    return fn
