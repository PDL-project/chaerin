def put_apple_in_fridge_and_switch_off_light(robots):
    # Task: Put Apple in Fridge
    action_queue.append({'action': 'Start Task: Put Apple in Fridge'})
    GoToObject(robots[0], 'Apple')
    PickupObject(robots[0], 'Apple')
    GoToObject(robots[0], 'Fridge')
    PutObject(robots[0], 'Apple', 'Fridge')

    # Task: Switch off the Light
    action_queue.append({'action': 'Start Task: Switch off the Light'})
    GoToObject(robots[0], 'LightSwitch')
    SwitchOff(robots[0], 'Light')

# Threading setup
task1_thread = threading.Thread(target=put_apple_in_fridge_and_switch_off_light, args=(robots,))
task1_thread.start()
task1_thread.join()

# Action queue and completion
action_queue.append({'action': 'Done'})
task_over = True
time.sleep(5)