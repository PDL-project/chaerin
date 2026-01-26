It seems like you have provided multiple task descriptions for robotic tasks. I'll focus on the last one: "Put apple in fridge and switch off the light".

Here's a step-by-step decomposition of this task:

1. **Get Apple**:
	* GoToObject: Robot goes to the apple.
	* Parameters: ?robot, ?apple
	* Preconditions: (not (inaction ?robot))
	* Effects: (at ?robot ?apple), (not (inaction ?robot))
2. **Pick up Apple**:
	* PickupObject: Robot picks up the apple.
	* Parameters: ?robot, ?apple, ?location (where apple is initially located)
	* Preconditions: (at-location ?apple ?location), (at ?robot ?location), (not (inaction ?robot))
	* Effects: (holding ?robot ?apple), (not (inaction ?robot))
3. **Go to Fridge**:
	* GoToObject: Robot goes to the fridge.
	* Parameters: ?robot, ?fridge
	* Preconditions: (not (inaction ?robot))
	* Effects: (at ?robot ?fridge), (not (inaction ?robot))
4. **Put Apple in Fridge**:
	* PutObject: Robot puts the apple in the fridge.
	* Parameters: ?robot, ?apple, ?fridge
	* Preconditions: (holding ?robot ?apple), (at ?robot ?fridge), (not (inaction ?robot))
	* Effects: (at-location ?apple ?fridge), (not (holding ?robot ?apple)), (not (inaction ?robot))
5. **Go to Light Switch**:
	* GoToObject: Robot goes to the light switch.
	* Parameters: ?robot, ?lightSwitch
	* Preconditions: (not (inaction ?robot))
	* Effects: (at ?robot ?lightSwitch), (not (inaction ?robot))
6. **Switch off Light**:
	* SwitchObject: Robot switches off the light.
	* Parameters: ?robot, ?lightSwitch
	* Preconditions: (at ?robot ?lightSwitch), (not (inaction ?robot))
	* Effects: (switched-off ?lightSwitch), (not (inaction ?robot))

This decomposition assumes that the robot has already been initialized and is ready to perform tasks. The preconditions for each step ensure that the robot is in the correct location or holding the necessary object before performing the action.