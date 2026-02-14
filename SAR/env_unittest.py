import unittest
import argparse
from env import SAREnv

class EnvTest(unittest.TestCase):
    def init(self, num_agents=2, scene=1, seed=42):
        env=SAREnv(num_agents=num_agents, scene=scene, seed=seed)
        env.reset()
        return env

    def test_scenes(self):
        env=self.init(num_agents=2, scene=1)

    def test_parsing(self):
        env=self.init(2)
        """
        Formatting for actions, required action_args:

        'NavigateTo' : to_target_id (location),
        'Move' : direction, # Up, Down, Left, Right
        'Explore', # TODO

        'Carry' : from_target_id (person),
        'DropOff' : from_target_id (person), to_target_id (deposit)

        'StoreSupply' : to_target_id (deposit),
        'UseSupply' : from_target_id (fire), supply_type (on fire)
        'GetSupply' : from_target_id (deposit OR reservoir), supply_type (deposit)
        'ClearInventory',
        'NoOp',
        """
        kwargs=env.parse_action("NavigateTo(GreatFire)", 0)
        # print(kwargs)
        kwargs=env.parse_action("Move(Up)", 0)
        kwargs=env.parse_action("Move(Down)", 0)
        kwargs=env.parse_action("Move(Left)", 0)
        kwargs=env.parse_action("Move(Right)", 0)
        # print(kwargs)

        kwargs=env.parse_action("Carry(LostPersonTimmy)", 0)
        # print(kwargs)
        kwargs=env.parse_action("DropOff(LostPersonTimmy, DepositFacility)", 0)
        # print(kwargs)
        kwargs=env.parse_action("StoreSupply(DepositFacility)", 0)
        # print(kwargs)
        kwargs=env.parse_action("UseSupply(CaldorFire, Water)", 0)
        # print(kwargs)
        kwargs=env.parse_action("GetSupply(DepositFacility, Water)", 0)
        # print(kwargs)
        kwargs=env.parse_action("GetSupply(ReservoirUtah)", 0)
        # print(kwargs)
        kwargs=env.parse_action("ClearInventory()", 0)
        # print(kwargs)
        kwargs=env.parse_action("NoOp()", 0)
        # print(kwargs)
        kwargs=env.parse_action("Idle", 0)
        # print(kwargs)
        kwargs=env.parse_action("Done", 0)
        # print(kwargs)

    def test_texts(self):
        env=self.init(2)
        act_text=env.get_act_text(action="NavigateTo(Potato)", act_success=False, agent_idx=0, error_type="not_visible")
        act_text=env.get_act_text(action="NavigateTo(Potato)", act_success=False, agent_idx=0, error_type="not_interactable")
        act_text=env.get_act_text(action="StoreSupply(DepositFacility)", act_success=False, agent_idx=0, error_type="restricted_action")
        # print(act_text)

    def test_step_and_check(self):
        num_agents=2
        seed=3

        from core import AbsAgent
        AbsAgent.INVENTORY_CAPACITY=3

        env=self.init(num_agents=num_agents, seed=seed)

        all_actions=[
                ["UseSupply(GreatFire, Water)", "UseSupply(CaldorFire, Sand)"], # should fail, no such supply in inventory

                ["NavigateTo(ReservoirYork)", "NavigateTo(ReservoirUtah)"],
                ["GetSupply(ReservoirYork)", "GetSupply(ReservoirUtah)"],
                ["GetSupply(ReservoirYork)", "GetSupply(ReservoirUtah)"],
                ["GetSupply(ReservoirYork)", "GetSupply(ReservoirUtah)"],

                ["NavigateTo(GreatFire_Region_1)", "NavigateTo(CaldorFire_Region_1)"],
                ["UseSupply(GreatFire, Water)", "UseSupply(CaldorFire, Sand)"],

                ["Idle", "NavigateTo(CaldorFire_Region_2)"],
                ["UseSupply(GreatFire, Water)", "UseSupply(CaldorFire, Sand)"],

                # ["NavigateTo(GreatFire)", "NavigateTo(CaldorFire_Region_1)"],
                # ["UseSupply(GreatFire, Water)", "UseSupply(CaldorFire, Sand)"],

                ["Move(Left)", "Move(Right)"],
                ["Move(Left)", "Move(Right)"],
                ["Explore()", "Explore()"],
                ]
        # explore until we find the person
        for _ in range(20): all_actions.append(["Explore()", "Explore()"])
        all_actions+=[
                ["NavigateTo(LostPersonTimmy)", "NavigateTo(LostPersonTimmy)"],
                ["Carry(LostPersonTimmy)", "Carry(LostPersonTimmy)"],
                ["NavigateTo(DepositFacility)", "NavigateTo(DepositFacility)"],
                ["DropOff(DepositFacility, LostPersonTimmy)", "DropOff(DepositFacility, LostPersonTimmy)"]
                ]

        all_successes=[
                [False]*num_agents,
        ]
        for _ in range(len(all_actions)-1): all_successes.append([True]*num_agents)

        if verbose:
            # print("checker subtasks:", env.checker.subtasks)
            # print("checker coverage:", env.checker.coverage, end='\n\n')
            pass

        end_at=3
        for clock,actions in enumerate(all_actions):
            input_dict,successes=env.step(actions)

            if clock>len(all_actions)-(end_at+1) and render: env.render()

            # test
            if clock<len(all_successes) and "Explore()" not in actions:
                self.assertEqual(successes, all_successes[clock])

            # checker
            coverage = env.checker.get_coverage()
            transport_rate = env.checker.get_transport_rate()

            if verbose:
                print(clock, actions, input_dict)
                print()
                
                # print(clock, "completed subtasks:", env.checker.subtasks_completed)
                # print(clock, "coverage:", env.checker.coverage_completed)
                # print()
                pass

        if verbose:
            # print("final coverage:", coverage)
            # print("final transport rate:", transport_rate)
            pass

    def test_explore(self):
        num_agents=3
        seeds=[10*i for i in range(10)]
        explore_steps=4

        from core import AbsAgent
        AbsAgent.INVENTORY_CAPACITY=3

        # try for different seeds
        for seed in seeds:

            env=self.init(num_agents=num_agents, seed=seed)
            # print(env.object_dict['available_object_names'])

            # TODO: testing
            all_actions=[ ["Explore()"]*num_agents ] * explore_steps

            found_at=-1
            explores=0
            for clock,actions in enumerate(all_actions):
                input_dict,successes=env.step(actions)

                # if render: env.render()

                # checker
                coverage = env.checker.get_coverage()
                transport_rate = env.checker.get_transport_rate()

                # found timmy test
                if ('Explore()' in actions) and found_at == -1: explores+=1
                if 'LostPersonTimmy' in input_dict: found_at=explores # could be 0, if no explore was required (this should be rare)

            if verbose:
                if 'LostPersonTimmy' in input_dict:
                    print(seed, f"Found!!!!!!!!!! After {explores} steps!")
                else: print(seed, "No*************")

def main():
    global render
    global verbose

    render=False
    verbose=False
    unittest.main()

if __name__=="__main__":
    main()


