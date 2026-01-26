```pddl
(define (domain robot_tasks)
  (:requirements :durative-actions)
  
  (:durative-action goto
    :parameters (?robot - robot ?object - object)
    :duration (= ?duration 1)
    :condition (and 
                (at start (not (inaction ?robot)))
                (at start (not (at ?robot ?object)))
                (over all (not (inaction ?robot)))
              )
    :effect (and 
              (at end (at ?robot ?object))
              (at end (not (inaction ?robot)))
            )
  )

  (:durative-action pickup
    :parameters (?robot - robot ?object - object)
    :duration (= ?duration 1)
    :condition (and 
                (at start (not (inaction ?robot)))
                (at start (at-location ?object ?location))
                (at start (at ?robot ?location))
                (over all (not (inaction ?robot)))
              )
    :effect (and 
              (at end (holding ?robot ?object))
              (at end (not (inaction ?robot)))
            )
  )

  (:durative-action open
    :parameters (?robot - robot ?object - object)
    :duration (= ?duration 1)
    :condition (and 
                (at start (not (inaction ?robot)))
                (at start (at ?robot ?object))
                (over all (not (inaction ?robot)))
              )
    :effect (and 
              (at end (object-open ?robot ?object))
              (at end (not (inaction ?robot)))
            )
  )

  (:durative-action put
    :parameters (?robot - robot ?object - object ?location - location)
    :duration (= ?duration 1)
    :condition (and 
                (at start (holding ?robot ?object))
                (at start (at ?robot ?location))
                (at start (at-location ?object ?location))
                (over all (not (inaction ?robot)))
              )
    :effect (and 
              (at end (at-location ?object ?location))
              (at end (not (holding ?robot ?object)))
              (at end (not (inaction ?robot)))
            )
  )

  (:durative-action close
    :parameters (?robot - robot ?object - object)
    :duration (= ?duration 1)
    :condition (and 
                (at start (not (inaction ?robot)))
                (at start (at ?robot ?object))
                (over all (not (inaction ?robot)))
              )
    :effect (and 
              (at end (object-close ?robot ?object))
              (at end (not (inaction ?robot)))
            )
  )

  (:durative-action switchoff
    :parameters (?robot - robot ?object - object)
    :duration (= ?duration 1)
    :condition (and 
                (at start (not (inaction ?robot)))
                (at start (at ?robot ?object))
                (over all (not (inaction ?robot)))
              )
    :effect (and 
              (at end (switch-off ?robot ?object))
              (at end (not (inaction ?robot)))
            )
  )

  (:durative-action put_apple_in_fridge_and_switchoff_light
    :parameters (?robot1 - robot ?robot2 - robot ?apple - object ?fridge - object ?lightSwitch - object)
    :duration (= ?duration 1)
    :condition (and 
                (at start (not (inaction ?robot1)))
                (at start (not (inaction ?robot2)))
                (at start (not (at ?robot1 ?apple)))
                (at start (not (holding ?robot1 ?apple)))
                (at start (not (object-open ?robot1 ?fridge)))
                (at start (not (at ?robot2 ?lightSwitch)))
                (at start (object-on ?lightSwitch))
                (over all (not (inaction ?robot1)))
                (over all (not (inaction ?robot2)))
              )
    :effect (and 
              (at end (at ?robot1 ?apple))
              (at end (holding ?robot1 ?apple))
              (at end (object-open ?robot1 ?fridge))
              (at end (at-location ?apple ?fridge))
              (at end (not (holding ?robot1 ?apple)))
              (at end (at ?robot2 ?lightSwitch))
              (at end (switch-off ?robot2 ?lightSwitch))
              (at end (not (inaction ?robot1)))
              (at end (not (inaction ?robot2))
            )
  )
)
```