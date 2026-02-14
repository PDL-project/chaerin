import ai2thor.controller


kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

scenes = kitchens + living_rooms + bedrooms + bathrooms

controller = ai2thor.controller.Controller(width=800, height=800, scene="FloorPlan1")

# agentCount specifies the number of agents in a scene
# event = controller.step(dict(action='Initialize', gridSize=0.25, agentCount=2))
event = controller.step(
    dict(action="Initialize", gridSize=0.25, renderObjectImage=True, agentCount=1)
)

# print out agentIds
for e in event.events:
    print(e.metadata["agentId"])

# move the second agent ahead, agents are 0-indexed
event = controller.step(dict(action="MoveAhead", agentId=0))

# Get Reachable Positions
# Returns valid coordinates that the Agent can reach without colliding with the environment or Sim Objects. This can be used in tandem with Teleport to warp the Agent as needed. This is useful for things like randomizing the initial position of the agent without clipping into the environment.
# event = controller.step(dict(action='GetReachablePositions'))


event = controller.step(
    dict(action="PickupObject", objectId="Cup|+01.08|+00.90|-00.77")
)
det = event.instance_detections2D
list(det.instance_masks.keys())


def get_object_in_view(event, agent_id):
    detections = event.events[agent_id].instance_detections2D
    # detections = event.instance_detections2D
    return list(detections.instance_masks.keys())


# @adelmo - do I need to change this to add teleport action?
action_primitives = [
    "MoveAhead",
    "MoveBack",
    "MoveRight",
    "MoveLeft",
    "RotateRight",
    "RotateLeft",
    "LookUp",
    "LookDown",
    "Crouch",
    "Stand",
    "Teleport",  # need position, rotation, horizon, standing
    "GetReachablePositions",  # positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    "Done",
    "PickupObject",  # need objectId
    "PutObject",  # need objectId, receptacleObjectId
    "DropHandObject",
    "ThrowObject",  # need moveMagnitude
    "MoveHeldObjectAhead",  # need moveMagnitude
    "MoveHeldObjectBack",  # need moveMagnitude
    "MoveHeldObjectLeft",  # need moveMagnitude
    "MoveHeldObjectRight",  # need moveMagnitude
    "MoveHeldObjectUp",  # need moveMagnitude
    "MoveHeldObjectDown",  # need moveMagnitude
    "MoveHeldObject",  # need ahead, right, up
    "RotateHeldObject",  # need pitch, yaw, roll
    "DirectionalPush",  # need moveMagnitude, pushAngle, objectId
    "PushObject",  # need objectId
    "PullObject",  # need objectId
    "OpenObject",  # need objectId
    "CloseObject",  # need objectId
]
