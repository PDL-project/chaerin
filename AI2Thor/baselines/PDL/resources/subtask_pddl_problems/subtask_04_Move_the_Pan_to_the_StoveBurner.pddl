```pddl
(define (problem move-pan-to-stoveburner)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    pan - object
    stoveburner - object
    diningtable - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location pan diningtable)
    (at-location stoveburner floor)
    (not (holding robot1 pan))
  )

  (:goal (and
    (at-location pan stoveburner)
  ))
)
```