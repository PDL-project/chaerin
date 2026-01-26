# Threading setup
task1_thread = threading.Thread(target=execute_task, args=(robots,))
task1_thread.start()
task1_thread.join()

# Action queue and completion
action_queue.append({'action':'Done'})
task_over = True
time.sleep(5)