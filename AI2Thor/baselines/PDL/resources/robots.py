"""
PDL robot definitions.
"""

# Single template robot (all skills, high mass)
robot1 = {
    "name": "robot1",
    "skills": [
        "GoToObject",
        "OpenObject",
        "CloseObject",
        "BreakObject",
        "SliceObject",
        "SwitchOn",
        "SwitchOff",
        "CleanObject",
        "PickupObject",
        "PutObject",
        "PutObjectInFridge",
        "DropHandObject",
        "ThrowObject",
        "PushObject",
        "PullObject",
    ],
    "mass": 100,
}

# Duplicate the same robot config across IDs to mirror SmartLLM behavior.
# Keep length 28 to stay compatible with existing dataset robot IDs.
robots = [robot1 for _ in range(28)]
