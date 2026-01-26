(define (problem switch_off_light_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    LightSwitch - object
    CounterTop - object
  )
  (:init
    (at robot2 CounterTop)
    (at-location LightSwitch CounterTop)
    ; Removed (inaction robot2) to allow actions to be performed
  )
  (:goal
    (and
      (switch-off robot2 LightSwitch)
    )
  )
)