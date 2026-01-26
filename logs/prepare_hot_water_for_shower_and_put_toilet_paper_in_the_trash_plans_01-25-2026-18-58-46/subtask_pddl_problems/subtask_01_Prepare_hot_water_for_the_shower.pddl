(define (problem prepare-hot-water)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    faucet - object
    bathtub - object
    bathroom - object
  )

  (:init
    (at robot1 bathroom)              ; Robot starts in the bathroom
    (at-location faucet bathroom)     ; Faucet is located in the bathroom
    (at-location bathtub bathroom)    ; Bathtub is located in the bathroom
    (not (switch-on robot1 faucet))   ; Faucet is initially switched off
    (not (inaction robot1))           ; Robot is not inactive
  )

  (:goal
    (and
      (switch-on robot1 faucet)       ; The goal is to switch on the faucet
    )
  )
)