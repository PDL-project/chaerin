Based on the task decomposition for "Put apple in fridge and switch off the light", I will assign the tasks to the available robots.

**Task 1: Get Apple**

* Robot 1 has the necessary skills (GoToObject, PickupObject) and mass capacity.
* Assign Task 1 to Robot 1.

**Task 2: Pick up Apple**

* Since Robot 1 is already assigned to Task 1, it will also perform Task 2.

**Task 3: Go to Fridge**

* Robot 1 has the necessary skills (GoToObject) and mass capacity.
* Assign Task 3 to Robot 1.

**Task 4: Put Apple in Fridge**

* Since Robot 1 is already assigned to Tasks 1-3, it will also perform Task 4.

**Task 5: Go to Light Switch**

* Robot 2 has the necessary skills (GoToObject) and mass capacity.
* Assign Task 5 to Robot 2.

**Task 6: Switch off Light**

* Since Robot 2 is already assigned to Task 5, it will also perform Task 6.

The task allocation is as follows:

* Robot 1:
	+ Task 1: Get Apple
	+ Task 2: Pick up Apple
	+ Task 3: Go to Fridge
	+ Task 4: Put Apple in Fridge
* Robot 2:
	+ Task 5: Go to Light Switch
	+ Task 6: Switch off Light

This allocation ensures that each robot has the necessary skills and mass capacity for its assigned tasks. The tasks are also performed sequentially, with each robot completing one task before moving on to the next one.