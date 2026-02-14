(define (problem go_to_garbagecan_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    GarbageCan - object
    CounterTop - object
  )
  (:init
    (at robot2 CounterTop)
    (at-location GarbageCan CounterTop)
  )
  (:goal
    (at robot2 GarbageCan)
  )
)