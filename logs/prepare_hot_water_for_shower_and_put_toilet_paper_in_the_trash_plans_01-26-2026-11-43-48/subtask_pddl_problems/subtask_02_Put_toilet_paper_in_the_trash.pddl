(define (problem put-toilet-paper-in-trash)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    garbage_can - object
    toilet_paper - object
    bathroom - object
  )

  (:init
    (at robot1 bathroom)
    (at toilet_paper bathroom)
    (at garbage_can bathroom)
    (at-location toilet_paper bathroom)
    (at-location garbage_can bathroom)
    (not (holding robot1 toilet_paper))
  )

  (:goal
    (and
      (at-location toilet_paper garbage_can)
    )
  )
)