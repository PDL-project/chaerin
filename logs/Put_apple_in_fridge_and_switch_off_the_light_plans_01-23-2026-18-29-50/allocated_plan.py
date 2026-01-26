### Step 1: Task Decomposition and Subtasks

**Task:** Put apple in fridge and switch off the light.

- **Subtask 1:** Put apple in fridge  
  Skills required:  
  - GoToObject (to Apple)  
  - PickupObject (Apple)  
  - GoToObject (to Fridge)  
  - OpenObject (Fridge)  
  - PutObject (Apple into Fridge)  
  - CloseObject (Fridge)

- **Subtask 2:** Switch off the light  
  Skills required:  
  - GoToObject (LightSwitch)  
  - SwitchOff

These two subtasks are independent and can be done **in parallel**.

---

### Step 2: Analyze Robots' Skills and Mass Capacity

Robots available:

| Robot   | Skills                                                                                                   | Mass Capacity |
|---------|----------------------------------------------------------------------------------------------------------|---------------|
| robot1  | ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject'] | 100           |
| robot2  | Same as robot1                                                                                           | 100           |
| robot3  | Same as robot1                                                                                           | 100           |

- All robots have all the skills needed for both subtasks.
- The mass of Apple is ~0.2 kg, which is well within any robot's capacity.
- Fridge and LightSwitch have zero or negligible mass for manipulation.

---

### Step 3: Task Allocation Based on Skills, Mass, and Parallelism

Since both subtasks are independent, they can be assigned to different robots to run in parallel, minimizing total time.

**Option A: Use two robots**

- Assign **Subtask1 ("Put apple in fridge")** to **robot1**
- Assign **Subtask2 ("Switch off the light")** to **robot2**

This uses only two robots out of three available.

**Option B: Use one robot**

If only one robot is preferred or available, assign both subtasks sequentially to a single robot. But since we have three robots, Option A is better for efficiency.

---

### Final Solution:

- Assign **robot1** to perform Subtask1:
    - GoTo Apple → Pickup Apple → GoTo Fridge → Open Fridge → Put Apple inside → Close Fridge
- Assign **robot2** to perform Subtask2:
    - GoTo LightSwitch → SwitchOff Light

Robot3 remains free/unassigned but can serve as backup if needed.

---

### Summary:

| Subtask                  | Assigned Robot | Reasoning                                      |
|--------------------------|----------------|------------------------------------------------|
| Put apple in fridge      | robot1         | Has all required skills; sufficient mass capacity; allows parallel execution |
| Switch off the light     | robot2         | Has all required skills; sufficient mass capacity; allows parallel execution |

---

If you want, I can provide a detailed step-by-step action plan or PDDL plan for each subtask with assigned robots. Would you like that?