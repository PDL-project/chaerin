ai2thor_actions = [
  "GoToObject <robot><object>",
  "OpenObject <robot><object>",
  "CloseObject <robot><object>",
  "BreakObject <robot><object>",
  "SliceObject <robot><object><location>",   
  "SwitchOn <robot><object>",
  "SwitchOff <robot><object>",
  "CleanObject <robot><object>",
  "PickupObject <robot><object><location>", 
  "PutObject <robot><object><receptacleObject>",
  "PutObjectInFridge <robot><object><fridge>", 
  "DropHandObject <robot><object><location>",  
  "ThrowObject <robot><object><target>",       
  "PushObject <robot><object>",
  "PullObject <robot><object>",
]
ai2thor_actions = ", ".join(ai2thor_actions)
