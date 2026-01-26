(define (problem Apple)
  (:domain kitchen-domain)

  (:objects
   apple - object
   coffee-machine - object
   counter-top - object
   cup - object
   dining-table - object
   dish-sponge - object
   egg - object
   faucet - object
   floor - object
   fork - object
   fridge - object
   garbage-bag - object
   garbage-can - object
   kettle - object
   knife - object
   lettuce - object
   light-switch - object
   microwave - object
   mug - object
   pan - object
   pepper-shaker - object
   plate - object
   pot - object
   potato - object
   salt-shaker - object
   sink - object
   sink-basin - object
   soap-bottle - object
   spatula - object
   spoon - object
   stool - object
   stove-burner - object
   stove-knob - object
   toaster - object
   tomato - object
   window - object
   wine-bottle - object)

  (:init
   (at apple counter-top)
   (at coffee-machine counter-top)
   (at cup counter-top)
   (at dining-table floor)
   (at dish-sponge sink-basin)
   (at egg fridge)
   (at faucet sink)
   (at floor floor)
   (at fork plate)
   (at fridge floor)
   (at garbage-bag garbage-can)
   (at kettle stove-burner)
   (at knife counter-top)
   (at lettuce fridge)
   (at light-switch wall)
   (at microwave counter-top)
   (at mug cup)
   (at pan stove-burner)
   (at pepper-shaker counter-top)
   (at plate table)
   (at pot stove-burner)
   (at potato fridge)
   (at salt-shaker counter-top)
   (at sink floor)
   (at sink-basin sink)
   (at soap-bottle shower)
   (at spatula pan)
   (at spoon plate)
   (at stool chair)
   (at stove-burner counter-top)
   (at stove-knob stove-burner)
   (at toaster counter-top)
   (at tomato fridge)
   (at window wall)
   (at wine-bottle shelf))

  (:goal
   (and
    (sliced apple)
    (switch-on coffee-machine)
    (object-open cup)
    (at egg plate)
    (switch-off faucet)
    (cleaned floor)
    (holding fork)
    (switch-on microwave)
    (at mug table)
    (object-close pan)
    (break pepper-shaker)
    (sliced potato)
    (switch-on stove-burner)
    (switch-on toaster)
    (sliced tomato)
    (switch-off wine-bottle))))