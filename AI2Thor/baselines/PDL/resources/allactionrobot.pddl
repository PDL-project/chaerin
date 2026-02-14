(define (domain allactionrobot)
  ;; Use ADL to safely support forall/when and conditional effects
  (:requirements :strips :typing :negative-preconditions :adl :quantified-preconditions :action-costs)

  (:types robot object)

  (:predicates
    ;; robot pose / status
    (at ?r - robot ?x - object)
    (inaction ?r - robot)

    ;; manipulation
    (holding ?r - robot ?o - object)
    (at-location ?o - object ?loc - object)

    ;; switches / device state
    (switch-on ?r - robot ?x - object)
    (switch-off ?r - robot ?x - object)

    ;; open/close state (robot-indexed as you had)
    (object-open ?r - robot ?x - object)
    (object-close ?r - robot ?x - object)

    ;; break/slice/clean
    (broken ?o - object)
    (sliced ?o - object)
    (cleaned ?r - robot ?o - object)

    ;; fridge typing/state
    (is-fridge ?x - object)
    (fridge-open ?f - object)
  )

  (:functions (total-cost) - number)

  ;; ------------------------------------------------------------
  ;; Move: ensure robot is at exactly one object after move
  ;; ------------------------------------------------------------
  (:action GoToObject
    :parameters (?r - robot ?dest - object)
    :precondition (not (inaction ?r))
    :effect (and
      (at ?r ?dest)
      ;; remove any other (at ?r ?x) except dest
      (forall (?x - object)
        (when (and (at ?r ?x) (not (= ?x ?dest)))
          (not (at ?r ?x))
        )
      )
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  ;; ------------------------------------------------------------
  ;; Pickup / Put / Drop
  ;; NOTE: we unify semantics to "robot goes to the OBJECT" (at r o),
  ;;       because your pipeline uses GoToObject(robot, tomato).
  ;; ------------------------------------------------------------
  (:action PickupObject
    :parameters (?r - robot ?o - object ?loc - object)
    :precondition (and
      (at-location ?o ?loc)
      (at ?r ?o)              ;; IMPORTANT: robot at object, not at location
      (not (inaction ?r))
    )
    :effect (and
      (holding ?r ?o)
      (not (at-location ?o ?loc)) ;; object no longer at old location
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  ;; PutObject is ONLY for non-fridge locations
  ;; Requires receptacle to NOT be closed (openable must be opened first)
  (:action PutObject
    :parameters (?r - robot ?o - object ?loc - object)
    :precondition (and
      (holding ?r ?o)
      (at ?r ?loc)
      (not (inaction ?r))
      (not (is-fridge ?loc))
      (not (object-close ?r ?loc))
    )
    :effect (and
      (at-location ?o ?loc)
      (not (holding ?r ?o))
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  ;; Put into fridge requires fridge-open to be true
  (:action PutObjectInFridge
    :parameters (?r - robot ?o - object ?f - object)
    :precondition (and
      (holding ?r ?o)
      (at ?r ?f)
      (not (inaction ?r))
      (is-fridge ?f)
      (fridge-open ?f)
    )
    :effect (and
      (at-location ?o ?f)
      (not (holding ?r ?o))
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  ;; Drop: same as put, but no fridge restriction
  ;; Requires receptacle to NOT be closed (openable must be opened first)
  (:action DropHandObject
    :parameters (?r - robot ?o - object ?loc - object)
    :precondition (and
      (holding ?r ?o)
      (at ?r ?loc)
      (not (inaction ?r))
      (not (is-fridge ?loc))
      (not (object-close ?r ?loc))
    )
    :effect (and
      (at-location ?o ?loc)
      (not (holding ?r ?o))
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  ;; ------------------------------------------------------------
  ;; ThrowObject: THIS IS THE MAIN SOURCE OF "CHEAT" PLANS.
  ;; To avoid any weird shortcut, we DISALLOW throwing into fridges.
  ;; If you don't need ThrowObject at all, you can delete this action entirely.
  ;; ------------------------------------------------------------
  (:action ThrowObject
    :parameters (?r - robot ?o - object ?target - object)
    :precondition (and
      (holding ?r ?o)
      (not (inaction ?r))
      (not (is-fridge ?target))  ;; critical anti-cheat
    )
    :effect (and
      (at-location ?o ?target)
      (not (holding ?r ?o))
      (not (inaction ?r))
      (increase (total-cost) 10) ;; 높은 비용: 플래너가 PutObject를 선호하도록
    )
  )

  ;; ------------------------------------------------------------
  ;; Switch / Clean / Break / Slice
  ;; ------------------------------------------------------------
  (:action SwitchOn
    :parameters (?r - robot ?x - object)
    :precondition (and
      (not (inaction ?r))
      (at ?r ?x)
    )
    :effect (and
      (switch-on ?r ?x)
      (not (switch-off ?r ?x))
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  (:action SwitchOff
    :parameters (?r - robot ?x - object)
    :precondition (and
      (not (inaction ?r))
      (at ?r ?x)
    )
    :effect (and
      (switch-off ?r ?x)
      (not (switch-on ?r ?x))
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  (:action CleanObject
    :parameters (?r - robot ?o - object)
    :precondition (and
      (not (inaction ?r))
      (at ?r ?o)
    )
    :effect (and
      (cleaned ?r ?o)
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  (:action BreakObject
    :parameters (?r - robot ?o - object)
    :precondition (and
      (not (inaction ?r))
      (at ?r ?o)
    )
    :effect (and
      (broken ?o)
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  (:action SliceObject
    :parameters (?r - robot ?o - object ?loc - object)
    :precondition (and
      (at-location ?o ?loc)
      (at ?r ?o)            ;; unify with "robot at object"
      (not (inaction ?r))
    )
    :effect (and
      (sliced ?o)
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  ;; ------------------------------------------------------------
  ;; Open/Close (generic)
  ;; IMPORTANT: keep fridge-open consistent even if planner uses OpenObject/CloseObject
  ;; ------------------------------------------------------------
  (:action OpenObject
    :parameters (?r - robot ?x - object)
    :precondition (and
      (not (inaction ?r))
      (at ?r ?x)
    )
    :effect (and
      (object-open ?r ?x)
      (not (object-close ?r ?x))
      ;; if x is a fridge, opening it implies fridge-open
      (when (is-fridge ?x) (fridge-open ?x))
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  (:action CloseObject
    :parameters (?r - robot ?x - object)
    :precondition (and
      (not (inaction ?r))
      (at ?r ?x)
    )
    :effect (and
      (object-close ?r ?x)
      (not (object-open ?r ?x))
      ;; if x is a fridge, closing it implies not fridge-open
      (when (is-fridge ?x) (not (fridge-open ?x)))
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  ;; ------------------------------------------------------------
  ;; Push / Pull (no world change modeled)
  ;; ------------------------------------------------------------
  (:action PushObject
    :parameters (?r - robot ?o - object)
    :precondition (and
      (not (inaction ?r))
      (at ?r ?o)
    )
    :effect (and
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  (:action PullObject
    :parameters (?r - robot ?o - object)
    :precondition (and
      (not (inaction ?r))
      (at ?r ?o)
    )
    :effect (and
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  ;; ------------------------------------------------------------
  ;; Fridge specialized open/close
  ;; (Still useful and now consistent with OpenObject/CloseObject)
  ;; ------------------------------------------------------------
  (:action OpenFridge
    :parameters (?r - robot ?f - object)
    :precondition (and
      (not (inaction ?r))
      (at ?r ?f)
      (is-fridge ?f)
    )
    :effect (and
      (object-open ?r ?f)
      (not (object-close ?r ?f))
      (fridge-open ?f)
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )

  (:action CloseFridge
    :parameters (?r - robot ?f - object)
    :precondition (and
      (not (inaction ?r))
      (at ?r ?f)
      (is-fridge ?f)
      (fridge-open ?f)
    )
    :effect (and
      (object-close ?r ?f)
      (not (object-open ?r ?f))
      (not (fridge-open ?f))
      (not (inaction ?r))
      (increase (total-cost) 1)
    )
  )
)
