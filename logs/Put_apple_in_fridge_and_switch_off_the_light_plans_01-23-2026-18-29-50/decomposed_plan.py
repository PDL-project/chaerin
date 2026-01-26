Given the domain and objects, and the task description:

**Task:** Put apple in fridge and switch off the light.

---

### Step 1: Analyze the task and decompose subtasks

- Subtask 1: Put apple in fridge  
  Skills required: GoToObject, PickupObject, GoToObject (fridge), OpenObject (fridge), PutObject, CloseObject (fridge)

- Subtask 2: Switch off the light  
  Skills required: GoToObject (light switch), SwitchOff

These two subtasks are independent and can be done in parallel.

---

### Step 2: Identify objects involved

- Apple (object)
- Fridge (object, is-fridge)
- LightSwitch (object)
- Robot (choose any robot, e.g., robot1)

---

### Step 3: Plan actions for each subtask

---

#### Subtask 1: Put apple in fridge

1. **GoToObject(robot1, Apple)**  
   Preconditions: (not (inaction robot1))  
   Effects: (at robot1 Apple), (not (inaction robot1))

2. **PickupObject(robot1, Apple, AppleLocation)**  
   Preconditions: (at-location Apple AppleLocation), (at robot1 AppleLocation), (not (inaction robot1))  
   Effects: (holding robot1 Apple), (not (inaction robot1))

3. **GoToObject(robot1, Fridge)**  
   Preconditions: (not (inaction robot1))  
   Effects: (at robot1 Fridge), (not (inaction robot1))

4. **OpenObject(robot1, Fridge)**  
   Preconditions: (not (inaction robot1)), (at robot1 Fridge)  
   Effects: (object-open robot1 Fridge), (not (inaction robot1))

5. **PutObject(robot1, Apple, Fridge)**  
   Preconditions: (holding robot1 Apple), (at robot1 Fridge), (not (inaction robot1)), (or (not (is-fridge Fridge)) (> (fridge-state Fridge) 0))  
   Effects: (at-location Apple Fridge), (not (holding robot1 Apple)), (not (inaction robot1))

6. **CloseObject(robot1, Fridge)**  
   Preconditions: (not (inaction robot1)), (at robot1 Fridge)  
   Effects: (object-close robot1 Fridge), (not (inaction robot1))

---

#### Subtask 2: Switch off the light

1. **GoToObject(robot2, LightSwitch)**  
   Preconditions: (not (inaction robot2))  
   Effects: (at robot2 LightSwitch), (not (inaction robot2))

2. **Switchoff(robot2, LightSwitch)**  
   Preconditions: (not (inaction robot2)), (at robot2 LightSwitch)  
   Effects: (switch-off robot2 LightSwitch), (not (inaction robot2))

---

### Step 4: Summary of parallel execution

- Robot1 executes Subtask 1 (put apple in fridge)
- Robot2 executes Subtask 2 (switch off light)

---

### Optional: If only one robot is available, subtasks must be sequential.

---

If you want, I can generate a PDDL plan or a step-by-step action sequence for this task. Would you like that?