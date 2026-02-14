from env import SAREnv
from object_actions import get_closest_feasible_action

print("starting environment init...")
env=SAREnv(num_agents=3)
env.reset()
print("environment init-ed!")
print("\n")

while True:
    s=input("Input string to predict action from:")
    pred_action=get_closest_feasible_action(s, env.object_dict)
    print("Predicted action:", pred_action)
    print()
