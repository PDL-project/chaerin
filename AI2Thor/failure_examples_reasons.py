# ----- OLD ----
# OLD reason
FAILURE_REASON="""
If any robot's previous action failed, think why it failed and output the reason for failure. If the previous action was successful, output "None".
"""

# OLD reason
FAILURE_EXAMPLE=f"""
"failure reason": "{AGENT_NAMES[1]} failed to pickup lettuce earlier because it was far away from it so it needs to navigate closer to it",
"memory": "{AGENT_NAMES[0]} saw a refrigerator next to the table when {AGENT_NAMES[0]} was at co-ordinates (0.5, -1.5) and was facing north. {AGENT_NAMES[1]} saw a lettuce on the countertop next to the refrigerator when {AGENT_NAMES[1]} was at co-ordinates (1, 0) and was facing north.",
"reason": "{AGENT_NAMES[0]} can see an an apple and can pick it and place it on the countertop to progress towards completing the task. {AGENT_NAMES[1]} can see the lettuce but is far away from it to pick it up so it should go closer to it before picking it up.",
"subtask": "{AGENT_NAMES[0]} is currently trying to pick up the apple and {AGENT_NAMES[1]} is currently trying to get navigate to the lettuce",
"{AGENT_NAMES[0]}'s action": pick up Apple_1,
"{AGENT_NAMES[1]}'s action": navigate to Lettuce_1
"""

# ----- NEW OVERWRITE ----
# NEW hip reason
FAILURE_REASON="""
If any robot's previous action failed, use the previous history and your understanding of causality to think and rationalize about why it failed. Output the reason for failure and how to fix this in the next timestep. If the previous action was successful, output "None".
Common failure reasons to lookout for include: one agent blocking another so must move out of the way, agent can't see an object and must explore to find, agent is doing the same redundant actions, etc.
"""

# NEW reason ver 2 (slightly shorter average steps + better balance):
FAILURE_REASON="""
If any robot's previous action failed, use the previous history, your current knowledge of the room (i.e. what things are where), and your understanding of causality to think and rationalize about why the previous action failed. Output the reason for failure and how to fix this in the next timestep. If the previous action was successful, output "None".
Common failure reasons to lookout for include: one agent blocking another so must move out of the way, agent can't see an object or its destination and must explore (such as move, rotate, or look in a different direction) to find it, agent doing extraneous actions (such as drying objects when cleaning), etc. If the previous action was successful, output "None".
"""