(define (problem move-mug-to-coffeemachine)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    mug - object
    coffeeMachine - object
    diningTable - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location mug diningTable)
    (at-location coffeeMachine diningTable)
    (not (holding robot1 mug))
  )

  (:goal (and
    (at-location mug coffeeMachine)
  ))

  (:metric minimize (total-cost))
)