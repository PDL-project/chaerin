To accomplish the task of putting an apple in the fridge and switching off the light, we can decompose the task into two independent subtasks that can be executed in parallel:

1. **SubTask 1: Put the Apple in the Fridge**
   - **Skills Required**: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
   - **Steps**:
     1. **GoToObject**: Robot goes to the apple.
        - Parameters: `?robot`, `?apple`
        - Preconditions: `(not (inaction ?robot))`
        - Effects: `(at ?robot ?apple)`, `(not (inaction ?robot))`
     2. **PickupObject**: Robot picks up the apple.
        - Parameters: `?robot`, `?apple`, `?location` (where the apple is initially located)
        - Preconditions: `(at-location ?apple ?location)`, `(at ?robot ?location)`, `(not (inaction ?robot))`
        - Effects: `(holding ?robot ?apple)`, `(not (inaction ?robot))`
     3. **GoToObject**: Robot goes to the fridge.
        - Parameters: `?robot`, `?fridge`
        - Preconditions: `(not (inaction ?robot))`
        - Effects: `(at ?robot ?fridge)`, `(not (inaction ?robot))`
     4. **OpenObject**: Robot opens the fridge.
        - Parameters: `?robot`, `?fridge`
        - Preconditions: `(not (inaction ?robot))`, `(at ?robot ?fridge)`, `(is-fridge ?fridge)`
        - Effects: `(object-open ?robot ?fridge)`, `(increase (fridge-state ?fridge) 1)`, `(not (inaction ?robot))`
     5. **PutObject**: Robot puts the apple inside the fridge.
        - Parameters: `?robot`, `?apple`, `?fridge`
        - Preconditions: `(holding ?robot ?apple)`, `(at ?robot ?fridge)`, `(not (inaction ?robot))`
        - Effects: `(at-location ?apple ?fridge)`, `(not (holding ?robot ?apple))`, `(not (inaction ?robot))`
     6. **CloseObject**: Robot closes the fridge.
        - Parameters: `?robot`, `?fridge`
        - Preconditions: `(not (inaction ?robot))`, `(at ?robot ?fridge)`, `(object-open ?robot ?fridge)`, `(is-fridge ?fridge)`
        - Effects: `(object-close ?robot ?fridge)`, `(decrease (fridge-state ?fridge) 1)`, `(not (inaction ?robot))`

2. **SubTask 2: Switch Off the Light**
   - **Skills Required**: GoToObject, SwitchOff
   - **Steps**:
     1. **GoToObject**: Robot goes to the light switch.
        - Parameters: `?robot`, `?lightSwitch`
        - Preconditions: `(not (inaction ?robot))`
        - Effects: `(at ?robot ?lightSwitch)`, `(not (inaction ?robot))`
     2. **SwitchOff**: Robot switches off the light.
        - Parameters: `?robot`, `?lightSwitch`
        - Preconditions: `(not (inaction ?robot))`, `(at ?robot ?lightSwitch)`
        - Effects: `(switch-off ?robot ?lightSwitch)`, `(not (inaction ?robot))`

By executing these subtasks in parallel, the task of putting the apple in the fridge and switching off the light can be completed efficiently. Each robot can be assigned to a specific subtask, or a single robot can perform both tasks sequentially if necessary.