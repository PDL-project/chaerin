# parser = argparse.ArgumentParser()
# parser.add_argument("--task", type=int, default=0)
# parser.add_argument("--floorplan", type=int, default=0)
# parser.add_argument("--verbose", type=bool, default=False)
# args = parser.parse_args()


# class Args:
#     def __init__(self):
#         self.task = 0
#         self.floorplan = 0
#         self.verbose = False


# args = Args()

# # autoconfig
# auto = AutoConfig()
# auto.set_task(args.task)
# auto.set_floorplan(args.floorplan)
# timeout = auto.get_task_timeout()

# config = auto.config()

env = SmartLLMEnv(config)
d = env.reset(task=auto.task_string())
# logger
logger = Logger(env=env, baseline_name="SmartLLM_partial_obs")


# get reachable positions
reachable_positions = env.get_reachable_positions()


available_robots = []
for i in range(env.num_agents):
    robot_dict_copy = robots[i].copy()
    robot_dict_copy["name"] = "robot" + str(i + 1)
    available_robots.append(robot_dict_copy)


# get list of all visible objects from agents POV
def get_obs_pov():
    all_obs = []
    for agent_id in range(env.num_agents):
        obs = env.get_object_in_view(env.event, agent_id)
        # _, obs = env.generate_obs_text(agent_id)
        all_obs += obs
    objs = list(set(all_obs))
    return objs


objs = get_obs_pov()

action_queue = []

task_over = False


def exec_actions():

    while not task_over:
        if len(action_queue) > 0:
            try:
                act = action_queue[0]
                if act["action"] == "ObjectNavExpertAction":
                    multi_agent_event = env.controller.step(
                        dict(
                            action=act["action"],
                            position=act["position"],
                            agentId=act["agent_id"],
                        )
                    )
                    next_action = multi_agent_event.metadata["actionReturn"]

                    if next_action != None:
                        multi_agent_event = env.controller.step(
                            action=next_action,
                            agentId=act["agent_id"],
                            forceAction=True,
                        )

                elif act["action"] == "MoveAhead":
                    multi_agent_event = env.controller.step(
                        action="MoveAhead", agentId=act["agent_id"]
                    )

                elif act["action"] == "MoveBack":
                    multi_agent_event = env.controller.step(
                        action="MoveBack", agentId=act["agent_id"]
                    )

                elif act["action"] == "RotateLeft":
                    multi_agent_event = env.controller.step(
                        action="RotateLeft",
                        degrees=act["degrees"],
                        agentId=act["agent_id"],
                    )

                elif act["action"] == "RotateRight":
                    multi_agent_event = env.controller.step(
                        action="RotateRight",
                        degrees=act["degrees"],
                        agentId=act["agent_id"],
                    )

                elif act["action"] == "PickupObject":
                    multi_agent_event = env.controller.step(
                        action="PickupObject",
                        objectId=act["objectId"],
                        agentId=act["agent_id"],
                        forceAction=True,
                    )
                    # if multi_agent_event.metadata["errorMessage"] != "":
                    #     print(multi_agent_event.metadata["errorMessage"])
                    # else:
                    #     success_exec += 1
                    agent_id = act["agent_id"]
                    success = env.event.events[agent_id].metadata["lastActionSuccess"]
                    object_id = env.convert_object_id_to_readable(act["objectId"])
                    action = f"PickupObject({object_id})"
                    env.checker.perform_metric_check(
                        action, success, env.inventory[agent_id]
                    )

                elif act["action"] == "PutObject":
                    multi_agent_event = env.controller.step(
                        action="PutObject",
                        objectId=act["objectId"],
                        agentId=act["agent_id"],
                        forceAction=True,
                    )
                    # if multi_agent_event.metadata["errorMessage"] != "":
                    #     print(multi_agent_event.metadata["errorMessage"])
                    # else:
                    #     success_exec += 1
                    agent_id = act["agent_id"]
                    success = env.event.events[agent_id].metadata["lastActionSuccess"]
                    object_id = env.convert_object_id_to_readable(act["objectId"])
                    action = f"PutObject({object_id})"
                    env.checker.perform_metric_check(
                        action, success, env.inventory[agent_id]
                    )

                elif act["action"] == "ToggleObjectOn":
                    multi_agent_event = env.controller.step(
                        action="ToggleObjectOn",
                        objectId=act["objectId"],
                        agentId=act["agent_id"],
                        forceAction=True,
                    )
                    # if multi_agent_event.metadata["errorMessage"] != "":
                    #     print(multi_agent_event.metadata["errorMessage"])
                    # else:
                    #     success_exec += 1
                    agent_id = act["agent_id"]
                    success = env.event.events[agent_id].metadata["lastActionSuccess"]
                    object_id = env.convert_object_id_to_readable(act["objectId"])
                    action = f"ToggleObjectOn({object_id})"
                    env.checker.perform_metric_check(
                        action, success, env.inventory[agent_id]
                    )

                elif act["action"] == "ToggleObjectOff":
                    multi_agent_event = env.controller.step(
                        action="ToggleObjectOff",
                        objectId=act["objectId"],
                        agentId=act["agent_id"],
                        forceAction=True,
                    )
                    # if multi_agent_event.metadata["errorMessage"] != "":
                    #     print(multi_agent_event.metadata["errorMessage"])
                    # else:
                    #     success_exec += 1
                    agent_id = act["agent_id"]
                    success = env.event.events[agent_id].metadata["lastActionSuccess"]
                    object_id = env.convert_object_id_to_readable(act["objectId"])
                    action = f"ToggleObjectOff({object_id})"
                    env.checker.perform_metric_check(
                        action, success, env.inventory[agent_id]
                    )

                elif act["action"] == "OpenObject":
                    multi_agent_event = env.controller.step(
                        action="OpenObject",
                        objectId=act["objectId"],
                        agentId=act["agent_id"],
                        forceAction=True,
                    )
                    # if multi_agent_event.metadata["errorMessage"] != "":
                    #     print(multi_agent_event.metadata["errorMessage"])
                    # else:
                    #     success_exec += 1
                    agent_id = act["agent_id"]
                    success = env.event.events[agent_id].metadata["lastActionSuccess"]
                    object_id = env.convert_object_id_to_readable(act["objectId"])
                    action = f"OpenObject({object_id})"
                    env.checker.perform_metric_check(
                        action, success, env.inventory[agent_id]
                    )

                elif act["action"] == "CloseObject":
                    multi_agent_event = env.controller.step(
                        action="CloseObject",
                        objectId=act["objectId"],
                        agentId=act["agent_id"],
                        forceAction=True,
                    )
                    # if multi_agent_event.metadata["errorMessage"] != "":
                    #     print(multi_agent_event.metadata["errorMessage"])
                    # else:
                    #     success_exec += 1
                    agent_id = act["agent_id"]
                    success = env.event.events[agent_id].metadata["lastActionSuccess"]
                    object_id = env.convert_object_id_to_readable(act["objectId"])
                    action = f"CloseObject({object_id})"
                    env.checker.perform_metric_check(
                        action, success, env.inventory[agent_id]
                    )

                elif act["action"] == "SliceObject":
                    multi_agent_event = env.controller.step(
                        action="SliceObject",
                        objectId=act["objectId"],
                        agentId=act["agent_id"],
                        forceAction=True,
                    )
                    # if multi_agent_event.metadata["errorMessage"] != "":
                    #     print(multi_agent_event.metadata["errorMessage"])
                    # else:
                    #     success_exec += 1
                    agent_id = act["agent_id"]
                    success = env.event.events[agent_id].metadata["lastActionSuccess"]
                    object_id = env.convert_object_id_to_readable(act["objectId"])
                    action = f"SliceObject({object_id})"
                    env.checker.perform_metric_check(
                        action, success, env.inventory[agent_id]
                    )

                elif act["action"] == "ThrowObject":
                    multi_agent_event = env.controller.step(
                        action="ThrowObject",
                        agentId=act["agent_id"],
                        forceAction=True,
                    )
                    # if multi_agent_event.metadata["errorMessage"] != "":
                    #     print(multi_agent_event.metadata["errorMessage"])
                    # else:
                    #     success_exec += 1
                    agent_id = act["agent_id"]
                    success = env.event.events[agent_id].metadata["lastActionSuccess"]
                    object_id = env.convert_object_id_to_readable(act["objectId"])
                    action = f"ThrowObject({object_id})"
                    env.checker.perform_metric_check(
                        action, success, env.inventory[agent_id]
                    )

                elif act["action"] == "BreakObject":
                    multi_agent_event = env.controller.step(
                        action="BreakObject",
                        objectId=act["objectId"],
                        agentId=act["agent_id"],
                        forceAction=True,
                    )
                    # if multi_agent_event.metadata["errorMessage"] != "":
                    #     print(multi_agent_event.metadata["errorMessage"])
                    # else:
                    #     success_exec += 1
                    agent_id = act["agent_id"]
                    success = env.event.events[agent_id].metadata["lastActionSuccess"]
                    object_id = env.convert_object_id_to_readable(act["objectId"])
                    action = f"BreakObject({object_id})"
                    env.checker.perform_metric_check(
                        action, success, env.inventory[agent_id]
                    )

                elif act["action"] == "Done":
                    multi_agent_event = env.controller.step(action="Done")

            except Exception as e:
                print(e)

            action_queue.pop(0)


actions_thread = threading.Thread(target=exec_actions)
actions_thread.start()


def convert_robot_names2list(robots):
    final_robots = []
    if not isinstance(robots, list):
        # convert robot to a list
        for available_robot in available_robots:
            if robots == available_robot["name"]:
                final_robots.append(available_robot)
    else:
        for robot in robots:
            for available_robot in available_robots:
                if robot == available_robot["name"]:
                    final_robots.append(available_robot)
    return final_robots


def GoToObject(robots, dest_obj):

    robots = convert_robot_names2list(robots)
    num_agents = len(robots)
    # robots distance to the goal
    dist_goals = [10.0] * len(robots)
    prev_dist_goals = [10.0] * len(robots)
    count_since_update = [0] * len(robots)
    clost_node_location = [0] * len(robots)

    # list of objects in the scene and their centers
    objs = get_obs_pov()

    # TODO modify this
    objs_center = list(
        [
            obj["axisAlignedBoundingBox"]["center"]
            for obj in env.controller.last_event.metadata["objects"]
        ]
    )
    if "|" in dest_obj:
        # obj alredy given
        dest_obj_id = dest_obj
        pos_arr = dest_obj_id.split("|")
        dest_obj_center = {
            "x": float(pos_arr[1]),
            "y": float(pos_arr[2]),
            "z": float(pos_arr[3]),
        }
    else:
        found_object = False
        for idx, obj in enumerate(objs):
            match = re.match(dest_obj, obj)
            if match is not None:
                dest_obj_id = obj
                dest_obj_center = objs_center[idx]
                if dest_obj_center != {"x": 0.0, "y": 0.0, "z": 0.0}:
                    found_object = True
                    break  # find the first instance

        if not found_object:
            print(f"Object: {dest_obj} not found in the scene")

    print("Going to ", dest_obj_id, dest_obj_center)

    dest_obj_pos = [dest_obj_center["x"], dest_obj_center["y"], dest_obj_center["z"]]

    # closest reachable position for each robot
    # all robots cannot reach the same spot
    # differt close points needs to be found for each robot
    crp = closest_node(
        dest_obj_pos, reachable_positions, num_agents, clost_node_location
    )

    goal_thresh = 0.25
    # at least one robot is far away from the goal

    while all(d >= goal_thresh for d in dist_goals):
        print(dist_goals)
        for ia, robot in enumerate(robots):
            robot_name = robot["name"]
            agent_id = int(robot_name[-1]) - 1

            # get the pose of robot
            metadata = env.controller.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"],
            }

            prev_dist_goals[ia] = dist_goals[ia]  # store the previous distance to goal
            dist_goals[ia] = distance_pts(
                [location["x"], location["y"], location["z"]], crp[ia]
            )

            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            # print (ia, "Dist to Goal: ", dist_goals[ia], dist_del, clost_node_location[ia])
            if dist_del < 0.2:
                # robot did not move
                count_since_update[ia] += 1
            else:
                # robot moving
                count_since_update[ia] = 0

            if count_since_update[ia] < 8:
                action_queue.append(
                    {
                        "action": "ObjectNavExpertAction",
                        "position": dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]),
                        "agent_id": agent_id,
                    }
                )
            else:
                # updating goal
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(
                    dest_obj_pos,
                    reachable_positions,
                    num_agents,
                    clost_node_location,
                )

            # time.sleep(0.5)

    # align the robot once goal is reached
    # compute angle between robot heading and object
    metadata = env.controller.last_event.events[agent_id].metadata
    robot_location = {
        "x": metadata["agent"]["position"]["x"],
        "y": metadata["agent"]["position"]["y"],
        "z": metadata["agent"]["position"]["z"],
        "rotation": metadata["agent"]["rotation"]["y"],
        "horizon": metadata["agent"]["cameraHorizon"],
    }

    robot_object_vec = [
        dest_obj_pos[0] - robot_location["x"],
        dest_obj_pos[2] - robot_location["z"],
    ]
    y_axis = [0, 1]
    unit_y = y_axis / np.linalg.norm(y_axis)
    unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)

    angle = math.atan2(
        np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y)
    )
    angle = 360 * angle / (2 * np.pi)
    angle = (angle + 360) % 360
    rot_angle = angle - robot_location["rotation"]

    if rot_angle > 0:
        action_queue.append(
            {"action": "RotateRight", "degrees": abs(rot_angle), "agent_id": agent_id}
        )
    else:
        action_queue.append(
            {"action": "RotateLeft", "degrees": abs(rot_angle), "agent_id": agent_id}
        )

    print("Reached: ", dest_obj)
    # if dest_obj == "Cabinet" or dest_obj == "Fridge" or dest_obj == "CounterTop":
    #     recp_id = dest_obj_id


def PickupObject(robots, pick_obj):
    robots = convert_robot_names2list(robots)
    # if not isinstance(robots, list):
    #     # convert robot to a list
    #     robots = [robots]
    no_agents = len(robots)
    # robots distance to the goal
    for idx in range(no_agents):
        robot = robots[idx]
        print("Picking: ", pick_obj)
        robot_name = robot["name"]
        agent_id = int(robot_name[-1]) - 1
        # list of objects in the scene and their centers
        objs = get_obs_pov()
        found_object = False
        for idx, obj in enumerate(objs):
            match = re.match(pick_obj, obj)
            if match is not None:
                pick_obj_id = obj
                found_object = True
                break
        if not found_object:
            print(f"Object: {pick_obj} not found in the scene")
        print("Picking Up ", pick_obj_id)  # , dest_obj_center)
        action_queue.append(
            {"action": "PickupObject", "objectId": pick_obj_id, "agent_id": agent_id}
        )
        # time.sleep(1)


def PutObject(robot, put_obj, recp):
    robots = convert_robot_names2list(robot)
    robot = robots[0]
    robot_name = robot["name"]
    agent_id = int(robot_name[-1]) - 1
    objs = get_obs_pov()
    found_object = False
    for idx, obj in enumerate(objs):
        match = re.match(recp, obj)
        if match is not None:
            recp_obj_id = obj
            found_object = True
            break
    if not found_object:
        print(f"Object: {recp} not found in the scene")

    action_queue.append(
        {"action": "PutObject", "objectId": recp_obj_id, "agent_id": agent_id}
    )
    # time.sleep(1)


def SwitchOn(robot, sw_obj):
    print("Switching On: ", sw_obj)
    robots = convert_robot_names2list(robot)
    robot = robots[0]
    robot_name = robot["name"]
    agent_id = int(robot_name[-1]) - 1
    objs = get_obs_pov()

    # turn on all stove burner
    if sw_obj == "StoveKnob":
        found_object = False
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                found_object = True
                # removing this as we feel this is cheating where low-level actions are chained
                # GoToObject(robot, sw_obj_id)
                # time.sleep(1)
                action_queue.append(
                    {
                        "action": "ToggleObjectOn",
                        "objectId": sw_obj_id,
                        "agent_id": agent_id,
                    }
                )
                # time.sleep(0.1)
        if not found_object:
            print(f"Object: {sw_obj} not found in the scene")

    # all objects apart from Stove Burner
    else:
        found_object = False
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                found_object = True
                break  # find the first instance
        if not found_object:
            print(f"Object: {sw_obj} not found in the scene")
        # GoToObject(robot, sw_obj_id)
        # time.sleep(1)
        action_queue.append(
            {"action": "ToggleObjectOn", "objectId": sw_obj_id, "agent_id": agent_id}
        )
        # time.sleep(1)


def SwitchOff(robot, sw_obj):
    print("Switching Off: ", sw_obj)
    robots = convert_robot_names2list(robot)
    robot = robots[0]
    robot_name = robot["name"]
    agent_id = int(robot_name[-1]) - 1
    objs = get_obs_pov()

    # turn on all stove burner
    if sw_obj == "StoveKnob":
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                action_queue.append(
                    {
                        "action": "ToggleObjectOff",
                        "objectId": sw_obj_id,
                        "agent_id": agent_id,
                    }
                )
                # time.sleep(0.1)

    # all objects apart from Stove Burner
    else:
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break  # find the first instance
        # GoToObject(robot, sw_obj_id)
        # time.sleep(1)
        action_queue.append(
            {"action": "ToggleObjectOff", "objectId": sw_obj_id, "agent_id": agent_id}
        )
        # time.sleep(1)


def OpenObject(robot, sw_obj):
    robots = convert_robot_names2list(robot)
    robot = robots[0]
    robot_name = robot["name"]
    agent_id = int(robot_name[-1]) - 1
    objs = get_obs_pov()

    found_object = False
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            found_object = True
            break  # find the first instance

    if not found_object:
        print(f"Object: {sw_obj} not found in the scene")

    # GoToObject(robot, sw_obj_id)
    # time.sleep(1)
    action_queue.append(
        {"action": "OpenObject", "objectId": sw_obj_id, "agent_id": agent_id}
    )
    # time.sleep(1)


def CloseObject(robot, sw_obj):
    robots = convert_robot_names2list(robot)
    robot = robots[0]
    robot_name = robot["name"]
    agent_id = int(robot_name[-1]) - 1
    objs = get_obs_pov()

    found_object = False
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            found_object = True
            break  # find the first instance
    if not found_object:
        print(f"Object: {sw_obj} not found in the scene")

    # GoToObject(robot, sw_obj_id)
    # time.sleep(1)

    action_queue.append(
        {"action": "CloseObject", "objectId": sw_obj_id, "agent_id": agent_id}
    )

    # time.sleep(1)


def BreakObject(robot, sw_obj):
    robots = convert_robot_names2list(robot)
    robot = robots[0]
    robot_name = robot["name"]
    agent_id = int(robot_name[-1]) - 1
    objs = get_obs_pov()

    found_object = False
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            found_object = True
            break  # find the first instance
    if not found_object:
        print(f"Object: {sw_obj} not found in the scene")

    # GoToObject(robot, sw_obj_id)
    # time.sleep(1)
    action_queue.append(
        {"action": "BreakObject", "objectId": sw_obj_id, "agent_id": agent_id}
    )
    # time.sleep(1)


def SliceObject(robot, sw_obj):
    robots = convert_robot_names2list(robot)
    robot = robots[0]

    print("Slicing: ", sw_obj)
    robot_name = robot["name"]
    agent_id = int(robot_name[-1]) - 1
    objs = get_obs_pov()
    # objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    found_object = False
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            found_object = True
            break  # find the first instance
    if not found_object:
        print(f"Object: {sw_obj} not found in the scene")
    # GoToObject(robot, sw_obj_id)
    # time.sleep(1)
    action_queue.append(
        {"action": "SliceObject", "objectId": sw_obj_id, "agent_id": agent_id}
    )
    # time.sleep(1)


def CleanObject(robot, sw_obj):
    robots = convert_robot_names2list(robot)
    robot = robots[0]

    robot_name = robot["name"]
    agent_id = int(robot_name[-1]) - 1
    objs = get_obs_pov()

    found_object = False
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            found_object = True
            break  # find the first instance
    if not found_object:
        print(f"Object: {sw_obj} not found in the scene")

    # GoToObject(robot, sw_obj_id)
    # time.sleep(1)
    action_queue.append(
        {"action": "CleanObject", "objectId": sw_obj_id, "agent_id": agent_id}
    )
    # time.sleep(1)


def ThrowObject(robot, sw_obj):
    robots = convert_robot_names2list(robot)
    robot = robots[0]

    robot_name = robot["name"]
    agent_id = int(robot_name[-1]) - 1
    objs = get_obs_pov()
    # objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    found_object = False
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            found_object = True
            break  # find the first instance
    if not found_object:
        print(f"Object: {sw_obj} not found in the scene")
    action_queue.append(
        {"action": "ThrowObject", "objectId": sw_obj_id, "agent_id": agent_id}
    )
    # time.sleep(1)
