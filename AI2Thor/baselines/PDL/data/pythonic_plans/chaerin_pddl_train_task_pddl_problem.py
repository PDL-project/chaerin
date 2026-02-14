# EXAMPLE 1 - Task Description: Turn off the light

# pddl problem file
(define (problem switch-off-light)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    lightswitch - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location lightswitch floor)
    (switch-on robot1 lightswitch)
  )

  (:goal (and
    (switch-off robot1 lightswitch)
  ))
)


# EXAMPLE 2 - Task Description: Slice the Potato 

# problem file
(define (problem slice-potato)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    potato - object 
    knife - object
    cuttingboard - object
    diningtable - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))

    (at robot1 kitchen)

    (at-location potato diningtable)
    (at-location knife diningtable)
    (at-location cuttingboard diningtable)

    (not (holding robot1 potato))
    (not (holding robot1 knife))
  )

  (:goal (and
    (sliced potato)
  ))
)


# EXAMPLE 3 - Task Description: Put the Apple in the Fridge

# problem file
(define (problem put-apple-in-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    apple - object
    fridge - object
    kitchen - object
    countertop - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location apple countertop)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (not (holding robot1 apple))
  )

  (:goal (and
    (at-location apple fridge)
    (not (holding robot1 apple))
  ))
)


# EXAMPLE 4 - Task Description: Put the Fork in the Drawer
# NOTE: Drawers, cabinets, and other openable receptacles start CLOSED.
# The planner must OpenObject before PutObject, then CloseObject after.

# problem file
(define (problem put-fork-in-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    fork - object
    drawer - object
    diningtable - object
    floor - object
  )

  (:init
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
)