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
    (at start (at ?apple ?location))
    (at start (is-fridge ?fridge))
  )
  :effect (and
    (at start (at ?robot1 ?apple))
    (at start (holding ?robot1 ?apple))
    (at 2 (at ?robot1 ?fridge))
    (at 3 (object-open ?robot1 ?fridge))
    (at 4 (at ?apple ?fridge))
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
    (switch-off-light robot2 lightSwitch)
  )
)
```

### Explanation:

1. **Variable Correction**: The `at-location` and `variablelocation` issues have been corrected by using the appropriate variable names directly, such as `at` for location-related conditions and effects.

2. **Parallel Execution**: The plan is structured to execute both subtasks simultaneously, allowing efficient use of both robots' capabilities.