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

GoToObject(robot1, egg)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 egg)

PickupObject(robot1, egg, fridge)
   Preconditions:
     (at-location egg fridge)
     (at robot1 fridge)
     (not (inaction robot1))
   Effects:
     (holding robot1 egg)

GoToObject(robot1, fridge)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 fridge)

OpenFridge(robot1, fridge)
   Preconditions:
     (not (inaction robot1))
     (at robot1 fridge)
     (is-fridge fridge)
   Effects:
     (object-open robot1 fridge)
     (fridge-open fridge)

PutObjectInFridge(robot1, egg, fridge)
   Preconditions:
     (holding robot1 egg)
     (at robot1 fridge)
     (not (inaction robot1))
     (is-fridge fridge)
     (fridge-open fridge)
   Effects:
     (at-location egg fridge)
     (not (holding robot1 egg))
     (not (inaction robot1))

CloseFridge(robot1, fridge)
   Preconditions:
     (not (inaction robot1))
     (at robot1 fridge)
     (object-open robot1 fridge)
     (is-fridge fridge)
   Effects:
     (object-close robot1 fridge)
     (not (fridge-open fridge))

Goal condition: 
(at-location egg fridge)

# Initial condition analyze due to previous subtask:
#1. Robot not at apple location
#2. Robot not holding apple
#3. Robot not holding knife

# SubTask 2: Prepare Apple Slices. 
    Skills Required: GoToObject, PickupObject, SliceObject, PutObject
    Related Objects: apple(Location=counterTop), knife(Location=diningTable), pot(Location=stoveBurner)

GoToObject(robot1, apple)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 apple)

PickupObject(robot1, apple, counterTop)
   Preconditions:
     (at-location apple counterTop)
     (at robot1 apple)
     (not (inaction robot1))
   Effects:
     (holding robot1 apple)

GoToObject(robot1, cuttingboard)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 cuttingboard)

SliceObject(robot1, apple, cuttingboard)
   Preconditions:
     (at-location apple cuttingboard)
     (at robot1 cuttingboard)
     (not (inaction robot1))
   Effects:
     (sliced apple)

GoToObject(robot1, pot)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 pot)

PutObject(robot1, apple, pot)
   Preconditions:
     (holding robot1 apple)
     (at robot1 pot)
     (not (inaction robot1))
   Effects:
     (at-location apple pot)
     (not (holding robot1 apple))

Goal condition:
(sliced apple)
(at-location apple pot)

# Inital condition analyze due to previous subtask:
#1. Robot at pot location
#2. Fridge is Fridge, and initally closed
#3. Robot not holding pot initally.

# SubTask 3: Place the Pot with Apple Slices in the Fridge.
    Skills Required: GoToObject, PickupObject, PutObject, OpenObject, CloseObject
    Related Objects: pot(Location=stoveBurner), apple(Location=counterTop), fridge(Location=floor)

GoToObject(robot1, pot)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 pot)

PickupObject(robot1, pot, stoveBurner)
   Preconditions:
     (at-location pot stoveBurner)
     (at robot1 stoveBurner)
     (not (inaction robot1))
   Effects:
     (holding robot1 pot)

GoToObject(robot1, fridge)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 fridge)

OpenFridge(robot1, fridge)
   Preconditions:
     (not (inaction robot1))
     (at robot1 fridge)
     (is-fridge fridge)
   Effects:
     (object-open robot1 fridge)
     (fridge-open fridge)

PutObjectInFridge(robot1, pot, fridge)
   Preconditions:
     (holding robot1 pot)
     (at robot1 fridge)
     (not (inaction robot1))
     (is-fridge fridge)
     (fridge-open fridge)
   Effects:
     (at-location pot fridge)
     (not (holding robot1 pot))
     (not (inaction robot1))

CloseFridge(robot1, fridge)
   Preconditions:
     (not (inaction robot1))
     (at robot1 fridge)
     (object-open robot1 fridge)
     (is-fridge fridge)
   Effects:
     (object-close robot1 fridge)
     (not (fridge-open fridge))

Goal condition:
(at-location pot fridge)

# Task Put an Egg in the Fridge, and place a pot containing Apple slices into the refrigerator is done.