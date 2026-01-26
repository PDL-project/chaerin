# Define the task function
def execute_task(robots):
    # Task description
    GoToObject(robots[0], 'Apple')
    time.sleep(5)

    GoToObject(robots[1], 'LightSwitch')
    time.sleep(3)

    PickupObject(robots[0], 'Apple')
    time.sleep(2)

    SwitchOff(robots[1], 'LightSwitch')
    time.sleep(1)

    GoToObject(robots[0], 'Fridge')
    time.sleep(4)

    OpenObject(robots[0], 'Fridge')
    time.sleep(2)

    PutObject(robots[0], 'Apple', 'Fridge')
    time.sleep(2)

    CloseObject(robots[0], 'Fridge')
    time.sleep(2)

# Threading setup
task1_thread = threading.Thread(target=execute_task, args=(robots,))
task1_thread.start()
task1_thread.join()

# Action queue and completion
action_queue.append({'action':'Done'})
task_over = True
time.sleep(5)