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
    (inaction robot2)
  )
  (:goal
    (and
      (switch-off robot2 LightSwitch)
    )
  )
)