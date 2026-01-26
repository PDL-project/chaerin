import threading
import time

def put_apple_in_fridge(robots):
    # Task: Put the Apple in the Fridge
    # Go to the Apple.
    GoToObject(robots[0], 'Apple')
    # Pick up the Apple.
    PickupObject(robots[0], 'Apple')
    # Go to the Fridge.
    GoToObject(robots[0], 'Fridge')
    # Put the Apple in the Fridge.
    PutObject(robots[0], 'Apple', 'Fridge')

def switch_off_light(robots):
    # Task: Switch Off the Light
    # Go to the Light Switch.
    GoToObject(robots[1], 'LightSwitch')
    # Switch off the Light.
    SwitchOff(robots[1], 'LightSwitch')

# Threading setup
task1_thread = threading.Thread(target=put_apple_in_fridge, args=(robots,))
task2_thread = threading.Thread(target=switch_off_light, args=(robots,))

# Start executing both tasks in parallel
task1_thread.start()
task2_thread.start()

# Wait for both tasks to finish
task1_thread.join()
task2_thread.join()

# Action queue and completion
action_queue.append({'action': 'Done'})
task_over = True
time.sleep(5)