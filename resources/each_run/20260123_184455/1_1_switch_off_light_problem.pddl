(define (problem switch_off_light_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    LightSwitch - object
    counterTop - object
  )
  (:init
    (at robot2 counterTop)
    (at-location LightSwitch counterTop)
    (switch-on robot2 LightSwitch)
    (not(inaction robot2))
  )
  (:goal
    (and
      (switch-off robot2 LightSwitch)
    )
  )
)