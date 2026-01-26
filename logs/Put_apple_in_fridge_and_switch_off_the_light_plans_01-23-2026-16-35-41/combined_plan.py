To address the task of putting an apple in the fridge and switching off the light using two robots, we need to correct the subplans and then merge them into a single plan using timed durative actions. The goal is to execute these tasks in parallel, leveraging the capabilities of both robots. Below is the corrected and merged plan in PDDL format:

```lisp
; PDDL Plan for Parallel Execution of SubTasks

; SubTask 1: Put the Apple in the Fridge
; SubTask 2: Switch Off the Light

; Define the durative actions for each subtask

; Durative Action for SubTask 1: Put the Apple in the Fridge
(:durative-action put-apple-in-fridge
  :parameters (?robot1 - robot ?apple - object ?fridge - object)
  :duration (= ?duration 6)
  :condition (and
    (at start (not (inaction ?robot1)))
    (at start (at-location ?apple ?location))
    (at start (is-fridge ?fridge))
  )
  :effect (and
    (at start (at ?robot1 ?apple))
    (at start (holding ?robot1 ?apple))
    (at 2 (at ?robot1 ?fridge))
    (at 3 (object-open ?robot1 ?fridge))
    (at 4 (at-location ?apple ?fridge))
    (at 5 (not (holding ?robot1 ?apple)))
    (at end (object-close ?robot1 ?fridge))
  )
)

; Durative Action for SubTask 2: Switch Off the Light
(:durative-action switch-off-light
  :parameters (?robot2 - robot ?lightSwitch - object)
  :duration (= ?duration 2)
  :condition (and
    (at start (not (inaction ?robot2)))
  )
  :effect (and
    (at start (at ?robot2 ?lightSwitch))
    (at end (switch-off ?robot2 ?lightSwitch))
  )
)

; Plan Execution
; Execute both subtasks in parallel
(:plan
  (parallel
    (put-apple-in-fridge robot1 apple fridge)
    (switch-off-light robot2 lightswitch)
  )
)
```

### Explanation:

1. **Durative Actions**: Each subtask is defined as a durative action with a specified duration. The actions include conditions and effects that occur at different times during the action's execution.

2. **Parallel Execution**: The `:plan` section uses the `parallel` construct to execute both subtasks simultaneously. This allows `robot1` to handle the apple and fridge while `robot2` switches off the light.

3. **Variable Correction**: The `variablelocation` issue is addressed by using the variable itself, which includes the location information.

This plan ensures that both tasks are executed efficiently and in parallel, leveraging the capabilities of both robots.