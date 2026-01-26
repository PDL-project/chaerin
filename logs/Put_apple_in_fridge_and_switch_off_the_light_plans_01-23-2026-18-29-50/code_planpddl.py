```lisp
0.000: (gotoobject robot1 Apple) [duration: 5.0]
0.000: (gotoobject robot2 LightSwitch) [duration: 3.0]

5.000: (pickupobject robot1 Apple) [duration: 2.0]
3.000: (switchoff robot2 LightSwitch) [duration: 1.0]

7.000: (gotoobject robot1 Fridge) [duration: 4.0]

11.000: (openobject robot1 Fridge) [duration: 2.0]

13.000: (putobject robot1 Apple Fridge) [duration: 2.0]

15.000: (closeobject robot1 Fridge) [duration: 2.0]
```