(define (problem put_apple_in_fridge_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    apple - object
    fridge - object
    counterTop - object
    kitchenArea - object
    diningTable - object
  )
  (:init
    (at robot1 diningTable)
    (at-location apple counterTop)
    (at-location fridge kitchenArea)
    (not (inaction robot1))
  )
  (:goal
    (and
      (at-location apple fridge)
    )
  )
)