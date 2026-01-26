# SubTask 1: Put Apple in Fridge
# Initial conditions:
# 1. Robot not at apple location
# 2. Robot not holding apple
# 3. Fridge is fridge, and initially closed

GoToObject: Robot goes to the apple.
Parameters: ?robot, ?apple
Preconditions: (not (inaction ?robot))
Effects: (at ?robot ?apple), (not (inaction ?robot))

PickupObject: Robot picks up the apple.
Parameters: ?robot, ?apple, ?location (where apple is initially located)
Preconditions: (at-location ?apple ?location), (at ?robot ?location), (not (inaction ?robot))
Effects: (holding ?robot ?apple), (not (inaction ?robot))

GoToObject: Robot goes to the fridge.
Parameters: ?robot, ?fridge
Preconditions: (not (inaction ?robot))
Effects: (at ?robot ?fridge), (not (inaction ?robot))

OpenObject: Robot opens the fridge.
Parameters: ?robot, ?fridge
Preconditions: (not (inaction ?robot)), (at ?robot ?fridge)
Effects: (object-open ?robot ?fridge), (not (inaction ?robot))

PutObject: Robot puts the apple inside the fridge.
Parameters: ?robot, ?apple, ?fridge, ?location (inside the fridge)
Preconditions: (holding ?robot ?apple), (at ?robot ?fridge), (not (inaction ?robot))
Effects: (at-location ?apple ?fridge), (not (holding ?robot ?apple)), (not (inaction ?robot))

CloseObject: Robot closes the fridge.
Parameters: ?robot, ?fridge
Preconditions: (not (inaction ?robot)), (at ?robot ?fridge)
Effects: (object-close ?robot ?fridge), (not (inaction ?robot))

# SubTask 2: Switch Off the Light
# Initial conditions:
# 1. Robot not at light switch location
# 2. Light is switched on

GoToObject: Robot goes to the light switch.
Parameters: ?robot, ?lightSwitch
Preconditions: (not (inaction ?robot))
Effects: (at ?robot ?lightSwitch), (not (inaction ?robot))

Switchoff: Robot switches off the light.
Parameters: ?robot, ?lightSwitch
Preconditions: (not (inaction ?robot)), (at ?robot ?lightSwitch)
Effects: (switch-off ?robot ?lightSwitch), (not (inaction ?robot))

# Task Put apple in fridge and switch off the light is done.