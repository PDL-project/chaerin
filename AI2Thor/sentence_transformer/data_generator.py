import pandas as pd
import itertools
import functools
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
ai2thor_dir = os.path.dirname(script_dir)
file_path = os.path.join(ai2thor_dir, "objects_actions.tsv")

# @change
action="PutObject"
filename=action

cols=['Task', 'Function']
df = pd.DataFrame(columns=cols)
df_obj_act = pd.read_csv(file_path, sep="\t")

# options of object property
types=['Openable', 'Pickupable', 'On/Off', 'Receptacle', 'Fillable', 'Sliceable', 'Cookable', 'Breakable', 'Dirty', 'UsedUp']
# put object with specific properties that you want the example object to apply (like On/Off for ToggleObjectOn)
# @change
type_masks=['Pickupable']

# index range to put on object (i.e. object_1, object_2, etc)
# @change
index_range=2

applicable_objects=[]

def merge(x,y,fn):
    # element-wise merge
    return [fn(xx,yy) for xx,yy in zip(x,y)]

# this is doing 'or' for properties
mask=functools.reduce(lambda x,y: merge(x,y,lambda x,y:x or y), [df_obj_act[tm] == "yes" for tm in type_masks])
# this is doing 'and' for properties
mask=functools.reduce(lambda x,y: merge(x,y,lambda x,y:x and y), [df_obj_act[tm] == "yes" for tm in type_masks])
object_list = df_obj_act[mask]["Object Type"].tolist()

# smaller object_list
import numpy as np 
object_list=np.random.choice(object_list,12)

# @change
tasks = [
        'place object',
        'position item',
        'set down item',
        'place object',
        'position object',
        'set the object in it\'s place',
        ]

def numerate(obj, index):
    return f"{obj}_{index}"

def create_action(nobj):
    return f"{action}({nobj})"

def create_task(tsk, nobj):
    return f"{tsk} {nobj}"

for tsk, obj, idx in itertools.product(tasks, object_list, range(index_range)):
    numerated_obj=numerate(obj, idx)
    action_c = create_action(numerated_obj)
    task=create_task(tsk.capitalize(), numerated_obj)
    combinatorial_task_fn_pair = pd.DataFrame([[task, action_c]], columns=cols)

    df = pd.concat([df, combinatorial_task_fn_pair], ignore_index=True)

print(df)
df.to_csv(str(f"{filename}.csv"), index=False)
print(f"{filename}.csv")
