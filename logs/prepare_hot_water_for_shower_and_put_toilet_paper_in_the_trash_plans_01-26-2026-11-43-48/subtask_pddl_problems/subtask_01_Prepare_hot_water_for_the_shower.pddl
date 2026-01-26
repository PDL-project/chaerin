(define (problem prepare-hot-water)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    faucet - object
    bathtub - object
  )

  (:init
    (at robot1 faucet)           ; Robot is initially at the faucet
    (at-location faucet bathtub)  ; Faucet is located at the bathtub
    (not (inaction robot1))      ; Robot is not inactive
    (not (switch-on robot1 faucet))  ; Faucet is initially switched off
  )

  (:goal
    (and
      (switch-on robot1 faucet)  ; The goal is to switch on the faucet
    )
  )
)