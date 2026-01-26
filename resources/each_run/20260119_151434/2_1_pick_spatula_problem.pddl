(define (problem pick_spatula_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    spatula - object
    counterTop - object
  )
  (:init
    (at robot2 counterTop)
    (at-location spatula counterTop)
  )
  (:goal
    (and
      (holding robot2 spatula)
    )
  )
)