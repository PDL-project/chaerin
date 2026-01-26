(define (problem slice_potato_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    potato - object
    knife - object
    counterTop - object
  )
  (:init
    (at robot1 counterTop)
    (at-location potato counterTop)
    (at-location knife counterTop)
    (not (inaction robot1))
  )
  (:goal
    (and
      (sliced potato)
    )
  )
)