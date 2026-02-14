(define (problem slice_potato_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    potato - object
    knife - object
    counterTop - object
  )
  (:init
    (at robot2 counterTop)
    (at-location potato counterTop)
    (at-location knife counterTop)
    (not (inaction robot2))
  )
  (:goal
    (and
      (sliced potato)
    )
  )
)