(define (domain allactionrobot)
  (:requirements :strips :typing :negative-preconditions :adl)
  (:types robot object)

  (:predicates
    (at ?robot - robot ?object - object)
    (inaction ?robot - robot)
    (holding ?robot - robot ?object - object)
    (at-location ?object - object ?location - object)
    (switch-on ?robot - robot ?object - object)
    (switch-off ?robot - robot ?object - object)
    (object-open ?robot - robot ?object - object)
    (object-close ?robot - robot ?object - object)
    (break ?robot - robot ?object - object)
    (sliced ?object - object)
    (cleaned ?robot - robot ?object - object)
    (is-fridge ?object - object)
    (fridge-open ?fridge - object)
  )

  ;; Move to an object/location: ensure robot is at exactly one place after move
  (:action GoToObject
    :parameters (?robot - robot ?dest - object)
    :precondition (not (inaction ?robot))
    :effect (and
      (at ?robot ?dest)
      (forall (?x - object)
        (when (and (at ?robot ?x) (not (= ?x ?dest)))
          (not (at ?robot ?x))
        )
      )
      (not (inaction ?robot))
    )
  )

  (:action PickupObject
    :parameters (?robot - robot ?object - object ?location - object)
    :precondition (and
      (at-location ?object ?location)
      (at ?robot ?object)          ; 로봇이 object에 도착해야 함
      (not (inaction ?robot))
    )
    :effect (and
      (holding ?robot ?object)
      (not (inaction ?robot))
    )
  )

  ;; PutObject is ONLY for non-fridge locations
  (:action PutObject
    :parameters (?robot - robot ?object - object ?location - object)
    :precondition (and
      (holding ?robot ?object)
      (not (inaction ?robot))
      (at ?robot ?location)
      (not (is-fridge ?location))
    )
    :effect (and
      (at-location ?object ?location)
      (not (holding ?robot ?object))
      (not (inaction ?robot))
    )
  )

  ;; Put into fridge requires fridge to be open
  (:action PutObjectInFridge
    :parameters (?robot - robot ?object - object ?fridge - object)
    :precondition (and
      (holding ?robot ?object)
      (not (inaction ?robot))
      (at ?robot ?fridge)
      (is-fridge ?fridge)
      (fridge-open ?fridge)
    )
    :effect (and
      (at-location ?object ?fridge)
      (not (holding ?robot ?object))
      (not (inaction ?robot))
    )
  )

  (:action SwitchOn
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (switch-on ?robot ?object)
      (not (switch-off ?robot ?object))
    )
  )

  (:action Switchoff
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (switch-off ?robot ?object)
      (not (switch-on ?robot ?object))  ; 이거 추가
    )
  )

  (:action OpenObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (object-open ?robot ?object)
    )
  )

  (:action CloseObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (object-close ?robot ?object)
    )
  )

  (:action BreakObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (break ?robot ?object)
    )
  )

  (:action SliceObject
    :parameters (?robot - robot ?object - object ?location - object)
    :precondition (and
      (at-location ?object ?location)
      (at ?robot ?object)          ; object에 있어야 함
      (not (inaction ?robot))
    )
    :effect (and
      (not (inaction ?robot))
      (sliced ?object)
    )
  )

  (:action CleanObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (cleaned ?robot ?object)
    )
  )

  (:action OpenFridge
    :parameters (?robot - robot ?fridge - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?fridge)
      (is-fridge ?fridge)
    )
    :effect (and
      (not (inaction ?robot))
      (object-open ?robot ?fridge)
      (fridge-open ?fridge)
    )
  )

  (:action CloseFridge
    :parameters (?robot - robot ?fridge - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?fridge)
      (object-open ?robot ?fridge)
      (is-fridge ?fridge)
    )
    :effect (and
      (not (inaction ?robot))
      (object-close ?robot ?fridge)
      (not (fridge-open ?fridge))
    )
  )
)