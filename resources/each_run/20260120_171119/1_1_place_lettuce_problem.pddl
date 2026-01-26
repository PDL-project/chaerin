(define (problem place_lettuce_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    lettuce - object
    counterTop - object
    sink - object
  )
  (:init
    (at robot1 sink)
    (holding robot1 lettuce)
    (cleaned robot1 lettuce)
    (not (inaction robot1))
  )
  (:goal
    (at-location lettuce counterTop)
  )
)