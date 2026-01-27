# Task Description: Put an Egg in the Fridge, and place a pot containing Apple slices into the refrigerator.


# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analyze due to previous subtask:
#1. Robot not at egg location
#2. Robot not holding egg
#3. fridge is fridge, and initally closed

# SubTask 1: Put an Egg in the Fridge. 
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: egg(Location=fridge), fridge(Location=floor)

# Initial condition analyze due to previous subtask:
#1. Robot not at apple location
#2. Robot not holding apple
#3. Robot not holding knife

# SubTask 2: Prepare Apple Slices. 
    Skills Required: GoToObject, PickupObject, SliceObject, PutObject
    Related Objects: apple(Location=counterTop), knife(Location=diningTable), pot(Location=stoveBurner)

# Inital condition analyze due to previous subtask:
#1. Robot at pot location
#2. Fridge is Fridge, and initally closed
#3. Robot not holding pot initally.

# SubTask 3: Place the Pot with Apple Slices in the Fridge.
    Skills Required: GoToObject, PickupObject, PutObject, OpenObject, CloseObject
    Related Objects: pot(Location=stoveBurner), apple(Location=counterTop), fridge(Location=floor)

# Task Put an Egg in the Fridge, and place a pot containing Apple slices into the refrigerator is done.


# Task Description: Make a sandwich with sliced lettuce, sliced tomato, sliced bread and serve it on a washed plate.

# Initial condition analyze due to previous subtask:
#1. Robot not at any object location
#2. Robot not holding egg
#3. fridge is fridge, and initally closed

# GENERAL TASK DECOMPOSITION
# Decompose and parallelize subtasks where ever possible
# Independent subtasks:

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding lettuce.
# 2. Robot not at lettuce location.

# SubTask 1: Slice the Lettuce. 
    Skills Required: GoToObject, PickupObject, SliceObject, PutObject
    Related Objects: knife(Location=diningTable), lettuce(Location=diningTable)

# Initial Precondition analyze due to previous subtask:
#1. Robot not holding tomate
#2. Robot not at tomate location
#3. Robot not holding knife

# SubTask 2: Slice the Tomato. 
    Skills Required: GoToObject, PickupObject, SliceObject, PutObject
    Related Objects: knife(Location=diningTable), tomato(Location=counterTop)

#1. Robot not holding bread
#2. Robot not at  bread location.

# SubTask 3: Slice the Bread. 
    Skills Required: GoToObject, PickupObject, SliceObject, PutObject
    Related Objects: knife(Location=diningTable), bread(Location=diningTable)

#1. Robot not holding plate
#2. Robot not at plate location

# SubTask 4: Wash the Plate. 
    Skills Required: GoToObject, PickupObject, PutObject, SwitchOn, SwitchOff
    Related Object: plate(Location=counterTop), sink(Location=floor)

#1. Robot not holding bread, lettuce, or tomato
#2. Robot holding plate
#3. Robot not at bread, lettuce, or tomato location

# SubTask 5: Assemble the Sandwich. 
    Skills Required: GoToObject, PickupObject, PutObject
    Related Object: plate(Location=counterTop), bread(Location=diningTable), lettuce(Location=diningTable), tomato(Location=counterTop)

# Task Make a sandwich with sliced lettuce, sliced tomato, sliced bread and serve it on a washed plate is done.