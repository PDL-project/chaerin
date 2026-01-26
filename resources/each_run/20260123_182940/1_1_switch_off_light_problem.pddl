(define (problem switch_off_light_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    LightSwitch Floor - object
  )
  (:init
    (at robot2 Floor)
    (inaction robot2)
  )
  (:goal
    (switch-off robot2 LightSwitch)
  )
)