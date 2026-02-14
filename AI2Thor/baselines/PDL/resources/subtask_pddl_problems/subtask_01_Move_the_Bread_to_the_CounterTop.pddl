```lisp
(define (problem move-bread-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bread - object
    countertop - object
    diningtable - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location bread diningtable)
    (at-location countertop floor)
    (not (holding robot1 bread))
  )

  (:goal (and
    (at-location bread countertop)
  ))
)
```