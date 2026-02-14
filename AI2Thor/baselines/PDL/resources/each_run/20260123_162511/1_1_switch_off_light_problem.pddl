(define (problem switch_off_light_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    lightSwitch - object
    counterTop - object
  )
  (:init
    (at robot2 counterTop)
    (at-location lightSwitch counterTop)
    ;; Remove inaction since actions require not being inaction
  )
  (:goal
    (and
      (switch-off robot2 lightSwitch)
    )
  )
)