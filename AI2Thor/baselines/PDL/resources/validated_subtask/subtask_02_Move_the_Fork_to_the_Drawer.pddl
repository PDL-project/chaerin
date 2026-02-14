(define (problem move-fork-to-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    fork - object
    drawer - object
    diningtable - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location fork diningtable)
    (at-location drawer floor)
    (not (holding robot1 fork))
    (object-close robot1 drawer)
  )

  (:goal (and
    (at-location fork drawer)
    (object-close robot1 drawer)
  ))

  (:metric minimize (total-cost))
)