# Copyright 2022 Kaiyu Zheng
#
# Usage of this file is licensed under the MIT License.

"""
Navigation in ai2thor; pose transform is generic, but pose
convention follows ai2thor as below.

See project README; Poses in ai2thor:

 position (tuple): tuple (x, y, z); ai2thor uses (x, z) for robot base
 rotation (tuple): tuple (x, y, z); pitch, yaw, roll.
    Not doing quaternion because in ai2thor the mobile robot
    can only do two of the rotation axes so there's no problem using
    Euclidean.  Will use DEGREES. Will restrict the angles to be
    between 0 to 360 (same as ai2thor).

    yaw refers to rotation of the agent's body.
    pitch refers to rotation of the camera up and down.

 "Full pose" refers to (position, rotation), defined above
 "simplified pose" refers to (x, z, pitch, yaw)

 "action" refers to navigation action of the form (action_name, (forward, h_angle, v_angle))
 "action_delta" or "delta" refers to (forward, h_angle, v_angle)
"""

import math
import matplotlib.pyplot as plt
import time
from collections import deque
# @bug - added closest angle to import
from .utils import (
    PriorityQueue,
    euclidean_dist,
    to_radians,
    normalize_angles,
    roundany,
    floorany,
    closest,
    closest_angles,
    to_degrees,
)
from .constants import MOVEMENTS, MOVEMENT_PARAMS, H_ANGLES, V_ANGLES, GOAL_DISTANCE
from .object import thor_object_pose, thor_closest_object_of_type
from .agent import thor_reachable_positions


def convert_movement_to_action(movement, movement_params=MOVEMENT_PARAMS):
    """movement (str), a key in the constants.MOVEMENT_PARAMS dictionary
    Returns action tuple in the format:

    ("action name", (forward, h_angle, v_angle))"""
    if movement not in movement_params:
        raise ValueError(
            "Cannot convert movement {}." "We don't know about it.".format(movement)
        )
    params = movement_params[movement]
    forward, h_angle, v_angle = 0.0, 0.0, 0.0
    if "moveMagnitude" in params:
        forward = params["moveMagnitude"]
    if "degrees" in params and movement.startswith("Rotate"):
        if movement == "RotateLeft":
            h_angle = -params["degrees"]
        else:
            h_angle = params["degrees"]
    if "degrees" in params and movement.startswith("Look"):
        if movement == "LookUp":
            v_angle = -params["degrees"]
        else:
            v_angle = params["degrees"]
    return (movement, (forward, h_angle, v_angle))


def get_navigation_actions(movement_params=MOVEMENT_PARAMS, exclude=set()):
    return [
        convert_movement_to_action(movement, movement_params)
        for movement in movement_params
        if movement not in exclude
    ]


def _is_full_pose(robot_pose):
    return len(robot_pose) == 2 and len(robot_pose[0]) == 3 and len(robot_pose[1]) == 3


def _simplify_pose(robot_pose):
    if _is_full_pose(robot_pose):
        x, y, z = robot_pose[0]
        pitch, yaw, roll = robot_pose[1]
        return (x, z, pitch, yaw)
    return robot_pose


def _valid_pose(pose, reachable_positions):
    pose = _simplify_pose(pose)
    return (
        pose[:2] in reachable_positions
        and 0 <= pose[2] < 360.0
        and 0 <= pose[3] < 360.0
    )


def _move_by_vw(robot_pose, action_delta, grid_size=None, diagonal_ok=False):
    """
    Given 2D robot pose (x, z, pitch, yaw), and an action,
    which is (forward, h_angle, v_angle)
    Ai2Thor uses this model, as seen by MoveAhead, RotateLeft etc. actions.
    """
    rx, rz, pitch, yaw = robot_pose
    forward, h_angle, v_angle = action_delta
    new_yaw = yaw + h_angle  # angle (degrees)
    new_rx = rx + forward * math.sin(to_radians(new_yaw))
    new_rz = rz + forward * math.cos(to_radians(new_yaw))
    if grid_size is not None:
        if diagonal_ok:
            new_rx = roundany(new_rx, grid_size)
            new_rz = roundany(new_rz, grid_size)
        else:
            new_rx = floorany(new_rx, grid_size)
            new_rz = floorany(new_rz, grid_size)
    new_yaw = new_yaw % 360.0
    new_pitch = (pitch + v_angle) % 360.0
    return (new_rx, new_rz, new_yaw, new_pitch)


def _move_by_vw2d(robot_pose, action_delta, grid_size=None, diagonal_ok=False):
    """
    robot_pose (x, z, yaw); yaw is in DEGREES.
    action_delta (forward, angle)
    """
    rx, rz, rth = robot_pose
    forward, angle = action_delta
    new_rth = rth + angle
    new_rx = rx + forward * math.sin(to_radians(new_rth))
    new_rz = rz + forward * math.cos(to_radians(new_rth))
    if grid_size is not None:
        if diagonal_ok:
            new_rx = roundany(new_rx, grid_size)
            new_rz = roundany(new_rz, grid_size)
        else:
            new_rx = floorany(new_rx, grid_size)
            new_rz = floorany(new_rz, grid_size)
    new_rth = new_rth % 360.0
    return (new_rx, new_rz, new_rth)


def transform_pose(robot_pose, action, schema="vw", diagonal_ok=False, grid_size=None):
    """Transform pose of robot in 2D;
    This is a generic function, not specific to Thor.

    Args:
       robot_pose (tuple): Either 2d pose (x,y,yaw,pitch), or (x,y,yaw).
              or a tuple (position, rotation):
                  position (tuple): tuple (x, y, z)
                  rotation (tuple): tuple (x, y, z); pitch, yaw, roll.
       action:
              ("ActionName", delta), where delta is the change, format dependent on schema

       grid_size (float or None): If None, then will not
           snap the transformed x,y to grid.

       diagonal_ok (bool): True if it is ok to go diagonally,
           even though the traversed distance is longer than grid_size.

    Returns the transformed pose in the same form as input
    """
    action_name, delta = action
    if schema == "vw":
        x, z, pitch, yaw = _simplify_pose(robot_pose)
        new_pose = _move_by_vw(
            (x, z, pitch, yaw), delta, grid_size=grid_size, diagonal_ok=diagonal_ok
        )
    elif schema == "vw2d":
        x, z, yaw = robot_pose
        new_pose = _move_by_vw2d(
            (x, z, yaw), delta, grid_size=grid_size, diagonal_ok=diagonal_ok
        )
    else:
        raise ValueError("Unknown schema")

    if _is_full_pose(robot_pose):
        new_rx, new_rz, new_yaw, new_pitch = new_pose
        return (new_rx, robot_pose[0][1], new_rz), (
            new_pitch,
            new_yaw,
            robot_pose[1][2],
        )
    else:
        return new_pose


def _same_pose(pose1, pose2, tolerance=1e-4, angle_tolerance=5):
    """
    Returns true if pose1 and pose2 are of the same pose;
    Only cares about the coordinates that Ai2Thor cares about,
    which are x, z, pitch, yaw.

    pose1 and pose2 can either be full pose (i.e. (position, rotation)),
    or the simplified pose: (x, z, pitch, yaw)

    Args:
       tolerance (float): Euclidean distance tolerance
       angle_tolerance (float): Angular tolerance;
          Instead of relying on this tolerance, you
          should make sure the goal pose's rotation
          can be achieved exactly by taking the
          rotation actions.
    """
    if _is_full_pose(pose1):
        x1, _, z1 = pose1[0]
        pitch1, yaw1, _ = pose1[1]
    else:
        x1, z1, pitch1, yaw1 = pose1

    if _is_full_pose(pose2):
        x2, _, z2 = pose2[0]
        pitch2, yaw2, _ = pose2[1]
    else:
        x2, z2, pitch2, yaw2 = pose1

    return (
        euclidean_dist((x1, z1), (x2, z2)) <= tolerance
        # @tt - no pitch no more
        and abs(pitch1 - pitch2) <= angle_tolerance
        and abs(yaw1 - yaw2) <= angle_tolerance
    )


def _nav_heuristic(pose, goal):
    """Returns underestimate of the cost from pose to goal
    pose tuple(position, rotation); goal tuple(position, rotation)"""
    return euclidean_dist(pose[0], goal[0])


def _reconstruct_plan(comefrom, end_node, return_pose=False):
    """Returns the plan from start to end_node; The dictionary `comefrom` maps from node
    to parent node and the edge (i.e. action)."""
    plan = deque([])
    node = end_node
    while node in comefrom:
        parent_node, action = comefrom[node]
        if return_pose:
            plan.appendleft({"action": action, "next_pose": _simplify_pose(node)})
        else:
            plan.appendleft(action)
        node = parent_node
    return list(plan)


def _cost(action):
    """
    action is (movement_str, (forward, h_angle, v_angle))
    """
    forward, h_angle, v_angle = action[1]
    cost = 0
    if forward != 0:
        cost += 1
    if h_angle != 0:
        cost += 1
    if v_angle != 0:
        cost += 1
    return cost


def _round_pose(full_pose):
    x, y, z = full_pose[0]
    pitch, yaw, roll = full_pose[1]
    return (
        (round(x, 4), round(y, 4), round(z, 4)),
        (round(pitch, 4), round(yaw, 4), round(roll, 4)),
    )

#@nav - function to find the closest reachable position to target position
# this would allow the agent to go to exactly the closest rather than (possibly) some approximately close
def _closest_reachable_position_target(reachable_positions, target_position, tolerance=1e-4):
    # use L_2 norm
    xx,zz=target_position[0],target_position[2]
    r_position, _ = min(list(zip(reachable_positions, [(xx,zz)]*len(reachable_positions))), key=lambda t: euclidean_dist(t[0],t[1]))
    return r_position

def find_navigation_plan(
    start,
    goal,
    navigation_actions,
    reachable_positions,
    goal_distance=0.0,
    grid_size=None,
    angle_tolerance=5,
    diagonal_ok=False,
    return_pose=False,
    debug=False,
):
    """Returns a navigation plan as a list of navigation actions. Uses A*

    Recap of A*: A* selects the path that minimizes

    f(n)=g(n)+h(n)

    where n is the next node on the path, g(n) is the cost of the path from the
    start node to n, and h(n) is a heuristic function that estimates the cost of
    the cheapest path from n to the goal.  If the heuristic function is
    admissible, meaning that it never overestimates the actual cost to get to
    the goal, A* is guaranteed to return a least-cost path from start to goal.

    Args:
        start (tuple): position, rotation of the start
        goal (tuple): position, rotation of the goal n
        navigation_actions (list): list of navigation actions,
            represented as ("ActionName", (forward, h_angles, v_angles)),
        goal_distance (bool): acceptable minimum euclidean distance to the goal
        grid_size (float): size of the grid, typically 0.25. Only
            necessary if `diagonal_ok` is True
        diagonal_ok (bool): True if 'MoveAhead' can move
            the robot diagonally.
        return_pose (bool): True if return a list of {"action": <action>, "next_pose": <pose>} dicts
        debug (bool): If true, returns the expanded poses
    Returns:
        a list consisting of elements in `navigation_actions`
    """
    #   potential consequences - A* will not necessarily chose shortest path
    #   potential positive outcomes - (under reasonable assumptions) there exists a point which the agent can access AND interact with target object (that point is at least the closest one)
    if type(reachable_positions) != set:
        reachable_positions = set(reachable_positions)

    # The priority queue - create before conditional
    worklist = PriorityQueue()

    if debug:
        _expanded_poses = []
    
    # avoid NoneType errors - will return None if goal (or subcomponents) is/are None
    if goal is not None and not any([t is None for t in tuple(goal)]):
        # @nav - make target position equivalent to closest reachable position (main crux of problem - avoid ambiguity)
        goal_pos = _closest_reachable_position_target(reachable_positions, goal[0], tolerance=1e-4)
        #   add arbitrary z value to make it match dimensions (x,z)->(x,y,z) - shouldn't impact measurement of closeness (measure fn doesn't use y)
        goal_pos=(goal_pos[0], 0, goal_pos[1])
        goal = (goal_pos, goal[1])

        # Map angles in start and goal to be within 0 to 360 (see top comments)
        start_rotation = normalize_angles(start[1])
        goal_rotation = normalize_angles(goal[1])
        start = (start[0], start_rotation)
        goal = (goal[0], goal_rotation)

        worklist.push(start, _nav_heuristic(start, goal))
    else:
        # if None, don't add to worklist and return None (no path found)
        if debug:
            return None, _expanded_poses
        return None
    

    # cost[n] is the cost of the cheapest path from start to n currently known
    cost = {}
    cost[start] = 0

    # comefrom[n] is the node immediately preceding node n on the cheapeast path
    comefrom = {}

    # keep track of visited poses
    visited = set()

    while not worklist.isEmpty():
        current_pose = worklist.pop()
        if debug:
            _expanded_poses.append(current_pose)
        if _round_pose(current_pose) in visited:
            continue
        # @here - _same_pose hides the tolerancing by goal_distance (breaks everything)
        # @tt - changed to non-zero - zero now
        if _same_pose(
            current_pose, goal, tolerance=1e-5, angle_tolerance=angle_tolerance
        ):
            if debug:
                plan = _reconstruct_plan(comefrom, current_pose, return_pose=True)
                return plan, _expanded_poses
            else:
                return _reconstruct_plan(
                    comefrom, current_pose, return_pose=return_pose
                )

        for action in navigation_actions:
            next_pose = transform_pose(
                current_pose, action, grid_size=grid_size, diagonal_ok=diagonal_ok
            )
            if not _valid_pose(_round_pose(next_pose), reachable_positions):
                continue
            new_cost = cost[current_pose] + _cost(action)
            if new_cost < cost.get(next_pose, float("inf")):
                cost[next_pose] = new_cost
                worklist.push(
                    next_pose, cost[next_pose] + _nav_heuristic(next_pose, goal)
                )
                comefrom[next_pose] = (current_pose, action)

        visited.add(current_pose)

    # no path found
    if debug:
        return None, _expanded_poses
    else:
        return None


def get_shortest_path_to_object(
    controller, other_agents, object_id, start_position, start_rotation, **kwargs
):
    """
    Per this issue: https://github.com/allenai/ai2thor/issues/825
    Ai2thor's own method to compute shortest path is not desirable.
    I'm therefore writing my own using the algorithm above.
    It has almost identical call signature as the function of the same name
    under ai2thor.util.metric.

    Returns:
       If positions_only is True, returns a list of dict(x=,y=,z=) positions
       If return_plan is True, returns plan

        plan: List of (action_name, params) tuples representing actions.
            The params are specific to the action; For movements, it's
            (forward, pitch, yaw). For opening, it is an object id.
        poses: List of (dict(x=,y=,z=), dict(x=,y=,z=)) (position, rotation) tuples.
            where each pose at index i corresponds to the pose resulting from taking
            action i.
    """
    # Parameters
    v_angles = kwargs.get("v_angles", V_ANGLES)
    h_angles = kwargs.get("h_angles", H_ANGLES)
    movement_params = kwargs.get("movement_params", MOVEMENT_PARAMS)
    goal_distance = kwargs.get("goal_distance", GOAL_DISTANCE)
    diagonal_ok = kwargs.get("diagonal_ok", False)
    positions_only = kwargs.get("positions_only", False)
    return_plan = kwargs.get("return_plan", False)
    as_tuples = kwargs.get("as_tuples", False)

    # @tt - positions only is true
    # positions_only = True

    # @nav - change goal distance to be 0 (exact closest position)
    # goal_distance=controller.initialization_parameters["visibilityDistance"]
    goal_distance=0

    # reachable positions; must round them to grids otherwise ai2thor has a bug.
    grid_size = controller.initialization_parameters["gridSize"]
    reachable_positions = [
        tuple(map(lambda x: roundany(x, grid_size), pos))
        for pos in thor_reachable_positions(controller)
    ]

    # @multiagent - manually remove location of other agents from the list
    tuplize=lambda p : (p['x'], p['y'], p['z'])
    tuplize_xz=lambda p : (p['x'], p['z'])
    # do xz
    non_reachable_positions = [
        tuple(map(lambda x: roundany(x, grid_size), tuplize_xz(pos))) for pos in other_agents
    ]
    # add enclosure around these positions (agents are bigger than 1 grid position!)
    add=lambda p,delta: (p[0]+delta[0],p[1]+delta[1])
    sweep=[-1,0,1]
    for p in non_reachable_positions.copy():
        for deltax in sweep:
            for deltaz in sweep:
                # @cc - don't block diagonals
                # @tt - block diagonals now (hypothesis: agent is getting stuck?)
                # if abs(deltax)+abs(deltaz)>1:
                #   continue
                non_reachable_positions.append(add(p, (deltax*grid_size,deltaz*grid_size)))

    # @tt - subtlety, cannot remove the position of the agent (even if it lies inside the other's grid!)
    agent_position_formatted=(tuple(map(lambda x: roundany(x, grid_size), start_position)))
    agent_position_formatted=(agent_position_formatted[0],agent_position_formatted[2])
    non_reachable_positions=list(set(non_reachable_positions) - set([agent_position_formatted]))

    # @tt - add agent's own position into reachable positions!
    reachable_positions=list(set(reachable_positions+[agent_position_formatted]))
    # filter out
    reachable_positions=list(filter(lambda p : p not in non_reachable_positions, reachable_positions))

    # # added by @nsidn98 AI2Thor BUG: sometimes ai2thor doesn't consider agents' current
    # # position as reachable so we add it to the reachable positions
    # reachable_positions = reachable_positions + [
    #     tuple(map(lambda x: roundany(x, grid_size), start_position))
    # ]

    # Compute plan; Actually need to call the A* twice.
    # The first time determines the position the robot will end
    # up, if the target position is the one to reach,
    # which is then used to compute the pitch and yaw.
    # The second time then finds a complete plan that includes
    # reaching the pitch and yaw.
    #
    # Note that you can't just compute goal pitch and yaw some other
    # way, for example, using the closest position, because that
    # position may not be the end point on the shortest path.  Closest
    # reachable position to the target is not necessarily on the
    # Shortest path!!!!
    #
    # The algorithm is pretty fast so calling twice won't be an issue
    
    #@nav - note that it is not necessary to call A* twice as the assumption of indeterminacy of final position is nill 
    # yet, we call it anyways for now (avoid future bugs & breaking)
    # (remember) tolerance is 0 for the A* search
    target_position = thor_object_pose(controller, object_id, as_tuple=True)
    start_pose = (start_position, start_rotation)
    navigation_actions = get_navigation_actions(movement_params)

    params = dict(
        goal_distance=goal_distance,
        grid_size=grid_size,
        diagonal_ok=diagonal_ok,
        return_pose=True,
    )

    tentative_plan, expanded_poses = find_navigation_plan(
        start_pose,
        (target_position, start_rotation),
        navigation_actions,
        reachable_positions,
        debug=True,
        **params
    )
    # print('tentative_plan', str(tentative_plan), len(tentative_plan))
    # # For debugging purposes
    # plot_navigation_search_result(start_pose,
    #                               (target_position, start_rotation),
    #                               tentative_plan,
    #                               expanded_poses,
    #                               reachable_positions, grid_size, ax=None)
    #plt.show()

    if tentative_plan is None:
        return None, None
        # raise ValueError(
        #     "Plan not found from {} to {}".format(
        #         (start_position, start_rotation), object_id
        #     )
        # )

    # @bug - yaw, pitch calculations
    all_angles=[i for i in range(360)]
    _yaw,_pitch=None,None
    _yaw=_yaw_facing(start_position, target_position, all_angles)
    _pitch=_pitch_facing(start_position, target_position, v_angles)

    if len(tentative_plan) == 0:
        final_plan = []  # it appears that the robot is at where it should be
        # @bug - if robot is where it should be, it still should face position of the object
        #       have pitch be null

    else:
        # Get the last position
        last_pose = tentative_plan[-1]["next_pose"]
        last_position = (last_pose[0], start_position[1], last_pose[1])  # x,y,z

        # Get the true goal pose, with correct pitch and yaw
        goal_pitch = _pitch_facing(last_position, target_position, v_angles)
        goal_yaw = _yaw_facing(last_position, target_position, h_angles)
        goal_pose = (
            target_position,
            (goal_pitch, goal_yaw, start_rotation[2]),
        )  # roll is 0.0
        final_plan = find_navigation_plan(
            start_pose, goal_pose, navigation_actions, reachable_positions, **params
        )

        # @bug - calculations for the *precise* pitch and yaw
        _yaw=_yaw_facing(last_position, target_position, all_angles)
        _pitch=_pitch_facing(last_position, target_position, v_angles)

        # final_plan=tentative_plan

    poses = []
    actions = []
    # print(final_plan, tentative_plan)
    for step in final_plan:
        actions.append(step["action"])
        x, z, pitch, yaw = step["next_pose"]
        if positions_only:
            y = start_position[1]
            if as_tuples:
                poses.append((x, y, z))
            else:
                poses.append(dict(x=x, y=y, z=z))
        else:
            roll = start_rotation[2]
            if as_tuples:
                poses.append(((x, y, z), (pitch, yaw, roll)))
            else:
                poses.append(
                    (dict(x=x, y=start_position[1], z=z), dict(x=pitch, y=yaw, z=roll))
                )

    # @bug - some output hacking to pass information about the goal pitch & yaw into the function above this one
    if return_plan:
        _pitch=0 if _pitch is not None else _pitch
        poses.append((_pitch, _yaw))
        return poses, actions
    else:
        return poses


def get_shortest_path_to_object_type(controller, object_type, *args, **kwargs):
    """Similar to get_shortest_path_to_object except
    taking object_type as input."""
    obj = thor_closest_object_of_type(controller, object_type)
    return get_shortest_path_to_object(controller, obj["objectId"], *args, **kwargs)


def _pitch_facing(robot_position, target_position, angles):
    """
    Returns a pitch angle rotation such that
    if the robot is at `robot_position` and target is at
    `target_position`, the robot is facing the target.

    Args:
       robot_position (tuple): x, y, z position
       target_position (tuple): x, y, z position
       angles (list): Valid pitch angles (possible values for pitch
           in ai2thor agent rotation). Note that negative
           negative is up, positive is down
    Returns:
        .pitch angle between 0 - 360 degrees
    """
    angles = normalize_angles(angles)
    rx, ry, _ = robot_position
    tx, ty, _ = target_position
    # remember for pitch in thor, negative is up, positive is down
    pitch = (
        to_degrees(
            math.atan2(ry - ty, tx - rx)  # reverse y axis direction because of ^^
        )
        % 360
    )
    return closest(angles, pitch)


def _yaw_facing(robot_position, target_position, angles):
    """
    Returns a yaw angle rotation such that
    if the robot is at `robot_position` and target is at
    `target_position`, the robot is facing the target.

    Args:
       robot_position (tuple): x, y, z position
       target_position (tuple): x, y, z position
       angles (list): Valid yaw angles
    Returns:
        .yaw angle between 0 - 360 degrees
    """

    rx, _, rz = robot_position
    tx, _, tz = target_position
    yaw = to_degrees(math.atan2(tx - rx, tz - rz)) % 360

    # @bug - changed to fn from closest to closest_angles instead
    return closest_angles(angles, yaw)


# --------- Visualization ------------#
def plot_navigation_search_result(
    start, goal, plan, expanded_poses, reachable_positions, grid_size, ax=None
):
    """
    Plots the reachable positions (the grid map),
    the expanded poses during the search, and the plan.

    start, goal (tuple): position, rotation poses
    """

    def plot_map(ax, reachable_positions, start, goal):
        x = [p[0] for p in reachable_positions]
        z = [p[1] for p in reachable_positions]
        ax.scatter(x, z, s=300, c="gray", zorder=1)

        xs, _, zs = start[0]
        ax.scatter([xs], [zs], s=200, c="red", zorder=4)

        xg, _, zg = goal[0]
        ax.scatter([xg], [zg], s=200, c="green", zorder=4)

    ###start
    if ax is None:
        ax = plt.gca()

    x = [p[0] for p in reachable_positions]
    z = [p[1] for p in reachable_positions]
    plot_map(ax, reachable_positions, start, goal)

    ax.set_xlim(min(x) - grid_size, max(x) + grid_size)
    ax.set_ylim(min(z) - grid_size, max(z) + grid_size)

    x = [p[0][0] for p in expanded_poses]
    z = [p[0][2] for p in expanded_poses]
    c = [i for i in range(0, len(expanded_poses))]
    ax.scatter(x, z, s=120, c=c, zorder=2, cmap="bone")

    if plan is not None:
        for step in plan:
            x, z, _, _ = step["next_pose"]
            ax.scatter([x], [z], s=120, zorder=2, c="orange")


# --------- Metrics ------------#
def spl_ratio(li, pi, Si):
    """spl ratio for a single trial.
    li, pi, Si stands for shortest_path_length, actual_path_length, success for trial i.
    """
    if max(pi, li) > 0:
        pl_ratio = li / max(pi, li)
    else:
        pl_ratio = 1.0
    return float(Si) * pl_ratio


def compute_spl(episode_results):
    """
    Reference: https://arxiv.org/pdf/1807.06757.pdf

    Args:
        episode_results (list) List of tuples
            (shortest_path_distance, actual_path_distance, success),
             as required by the formula. `actual_path_distance` and
            `shortest_path_distance` are floats; success is boolean.
    Return:
        float: the SPL metric
    """
    # li, pi, Si stands for
    # shortest_path_distance, actual_path_distance, success for trial i.
    return sum(spl_ratio(li, pi, Si) for li, pi, Si in episode_results) / len(
        episode_results
    )
