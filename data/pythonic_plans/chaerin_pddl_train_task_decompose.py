# Task Description: Put an Egg in the Fridge, and place a pot containing Apple slices into the refrigerator.

# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:
# SubTask 1: Put an Egg in the Fridge. 
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: egg, fridge

# SubTask 2: Prepare Apple Slices. 
    Skills Required: GoToObject, PickupObject, SliceObject, PutObject
    Related Objects: apple, knife, pot

# SubTask 3: Place the Pot with Apple Slices in the Fridge.
    Skills Required: GoToObject, PickupObject, PutObject, OpenObject, CloseObject
    Related Objects: pot, apple, fridge

# Task Put an Egg in the Fridge, and place a pot containing Apple slices into the refrigerator is done.


# Task Description: Make a sandwich with sliced lettuce, sliced tomato, sliced bread and serve it on a washed plate.

# GENERAL TASK DECOMPOSITION
# Decompose and parallelize subtasks where ever possible
# Independent subtasks:
# SubTask 1: Slice the Lettuce. 
    Skills Required: GoToObject, PickupObject, SliceObject, PutObject
    Related Objects: knife, lettuce

# SubTask 2: Slice the Tomato. 
    Skills Required: GoToObject, PickupObject, SliceObject, PutObject
    Related Objects: knife, tomato

# SubTask 3: Slice the Bread. 
    Skills Required: GoToObject, PickupObject, SliceObject, PutObject
    Related Objects: knife, bread

# SubTask 4: Wash the Plate. 
    Skills Required: GoToObject, PickupObject, PutObject, SwitchOn, SwitchOff
    Related Object: plate, sink

# SubTask 5: Assemble the Sandwich. 
    Skills Required: GoToObject, PickupObject, PutObject
    Related Object: plate, sliced bread, sliced lettuce, sliced tomato

# Task Make a sandwich with sliced lettuce, sliced tomato, sliced bread and serve it on a washed plate is done.