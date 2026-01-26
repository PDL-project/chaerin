# pddl problem file
    # 0: SubTask 1: Turn off the light
(define (problem turn-off-desk-light)
  (:domain allactiondomain)

  (:objects
    robot1 - robot
    office - location
    desk - location
    desk_light - object
  )

  (:init
    (robot-at robot1 office)           ; Robot starts in the office
    (object-at desk_light desk)        ; Desk light is located at the desk
    (path-exists office desk)          ; Path exists from the office to the desk
    (reachable robot1 desk)            ; Robot can reach the desk
    (object-on desk_light)             ; Desk light is initially on
  )

  (:goal
    (and
      (robot-at robot1 desk)           ; Ensure the robot is at the desk
      (not (object-on desk_light))     ; The goal is for the desk light to be off
    )
  )
)


# EXAMPLE 2 - Task Description: Slice the Potato 
# GENERAL TASK DECOMPOSITION
# Independent subtasks:
# SubTask 1: Slice the Potato. (Skills Required: GoToObject, PickupObject, SliceObject, PutObject)

# problem file
(define (problem slice-potato-problem)
  (:domain allactiondomain)

  (:objects
    robot1 - robot
    kitchen - location
    potato1 - object
    knife1 - object   ; Assuming knife is needed and treated as an object
  )

  (:init
    (robot-at robot1 kitchen)          ; Robot is initially in the kitchen
    (object-at potato1 kitchen)        ; Potato is also in the kitchen
    (object-at knife1 kitchen)         ; Knife is also in the kitchen
    (whole potato1)                    ; Potato is whole and needs to be sliced
    (not (holding robot1 knife1))      ; Robot is not initially holding the knife
  )

  (:goal
    (and
      (sliced potato1)                 ; The goal is for the potato to be sliced
    )
  )
)


# EXAMPLE 3 - Task Description: Throw the fork and spoon in the trash
# GENERAL TASK DECOMPOSITION
# Independent subtasks:
# SubTask 1: Throw the Fork in the trash. (Skills Required: GoToObject, PickupObject, ThrowObject)
# SubTask 2: Throw the Spoon in the trash. (Skills Required: GoToObject, PickupObject, ThrowObject)

# problem file
(define (problem dispose-utensils-problem)
  (:domain allactiondomain)

  (:objects
    robot1 - robot
    kitchen - location
    trashcan - location  ; Assuming trashcan as a location for simplicity
    fork1 - object
    spoon1 - object
  )

  (:init
    (robot-at robot1 kitchen)          ; Robot starts in the kitchen
    (object-at fork1 kitchen)          ; Fork is located in the kitchen
    (object-at spoon1 kitchen)         ; Spoon is located in the kitchen
    (path-exists kitchen trashcan)     ; Path exists from kitchen to trashcan
    (reachable robot1 trashcan)        ; Robot can reach the trashcan
    (not (holding robot1 fork1))       ; Robot is not holding the fork initially
    (not (holding robot1 spoon1))      ; Robot is not holding the spoon initially
  )

  (:goal
    (and
      (robot-at robot1 trashcan)       ; Robot must be at the trashcan location
      (object-at fork1 trashcan)       ; The fork must be in the trashcan
      (object-at spoon1 trashcan)      ; The spoon must be in the trashcan
    )
  )
)