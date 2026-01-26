# Task function for putting apple in fridge and switching off the light
def put_apple_in_fridge_and_switch_off_light(robots):
    # Task: Put apple in fridge
    GoToObject(robots[0], 'Apple')
    PickupObject(robots[0], 'Apple')
    GoToObject(robots[0], 'Fridge')
    PutObject(robots[0], 'Apple', 'Fridge')

    # Task: Switch off the light
    GoToObject(robots[0], 'LightSwitch')
    SwitchOff(robots[0], 'LightSwitch')

# Threading setup
task_thread = threading.Thread(target=put_apple_in_fridge_and_switch_off_light, args=(robots,))
task_thread.start()
task_thread.join()

# Action queue and completion
action_queue.append({'action': 'Done'})
task_over = True
time.sleep(5)