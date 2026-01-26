(define (problem put_apple_in_fridge_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    apple - object
    fridge - object
    counterTop - object
    kitchen - object
  )
  (:init
    (at robot1 counterTop)
    (at-location apple counterTop)
    (at-location fridge kitchen)
  )
  (:goal
    (and
      (at-location apple fridge)
    )
  )
)