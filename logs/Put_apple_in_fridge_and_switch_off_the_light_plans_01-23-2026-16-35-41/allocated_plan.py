To accomplish the task of putting an apple in the fridge and switching off the light, we can decompose it into two independent subtasks that can be executed in parallel. Given that all three robots have identical skills and sufficient mass capacity, we can allocate these tasks efficiently.

### Task Decomposition

1. **SubTask 1: Put the Apple in the Fridge**
   - **Skills Required**: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
   - **Steps**:
     1. **GoToObject**: Robot goes to the apple.
     2. **PickupObject**: Robot picks up the apple.
     3. **GoToObject**: Robot goes to the fridge.
     4. **OpenObject**: Robot opens the fridge.
     5. **PutObject**: Robot puts the apple inside the fridge.
     6. **CloseObject**: Robot closes the fridge.

2. **SubTask 2: Switch Off the Light**
   - **Skills Required**: GoToObject, SwitchOff
   - **Steps**:
     1. **GoToObject**: Robot goes to the light switch.
     2. **SwitchOff**: Robot switches off the light.

### Task Allocation

Given that all robots have identical skills and sufficient mass capacity (100), any robot can perform either subtask independently:

- Assign `Robot1` to SubTask 1 (Put Apple in Fridge).
- Assign `Robot2` to SubTask 2 (Switch Off Light).

This allocation allows both tasks to be performed in parallel since they are independent of each other.

### Execution Plan

- `Robot1` will execute SubTask 1 by going through each step sequentially until completion.
- Simultaneously, `Robot2` will execute SubTask 2 by performing its steps sequentially until completion.

By utilizing two robots for these independent tasks, we ensure efficient task execution without any unnecessary delays or dependencies between subtasks. If needed or if one robot becomes unavailable, `Robot3` is also capable of performing either task due to its identical skill set and mass capacity as Robots 1 and 2.