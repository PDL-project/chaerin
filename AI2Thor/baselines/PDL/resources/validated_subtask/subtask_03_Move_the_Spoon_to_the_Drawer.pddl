(define (problem move-spoon-to-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    spoon - object
    drawer - object
    diningtable - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location spoon diningtable)
    (at-location drawer floor)
    (not (holding robot1 spoon))
    (object-close robot1 drawer)
  )

  (:goal (and
    (at-location spoon drawer)
    (object-close robot1 drawer)
  ))

  (:metric minimize (total-cost))
)