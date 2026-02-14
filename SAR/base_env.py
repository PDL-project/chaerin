import core
from core import Controller
from typing import List, Tuple, Dict
# for explore
import random, misc, math

class SARBaseEnv:
    """
    Contains useful functions for SAREnv.
    Avoid clutter in main class
    """ 

    """
    Controller docs:

    ------------------------------
    .step(action, action_args)
    action_args:
        'NavigateTo' : to_target_id (location),
        'Move' : direction, # Up, Down, Left, Right
        'Carry' : from_target_id (person),
        'DropOff' : from_target_id (person), to_target_id (deposit),
        'StoreSupply' : to_target_id (deposit),
        'UseSupply' : from_target_id (fire),
        'GetSupply' : from_target_id (deposit or reservoir),
        'ClearInventory',
    ------------------------------

    ------------------------------
    event={
        'success' : None,
        'global_obs' : None,
        'local_obs' : None,
        'visual_obs' : None, # non-existent for now
        'info' : '',
        }

        global_obs -> unordered list of formatted_objects
        local_obs -> dict. key<->(cardinal direction), value<->(list of formatted_objects @ delta)
            cardinal directions ->(Up,Down),(Left,Right) and all pairs from group 1 and 2
    ------------------------------

    ------------------------------
    Format of formatted_objects
        {
        # FOR ALL
        'position' : x,y position of the object,
        'name' : given name for object,
        'object_id' : id of the object,
        'type' : type of object (class)
        'collidable' : is object collidable,

        # SPECIFIC
        (if flammable) 'intensity' :  'none', 'low', 'medium', 'high',
        (if flammable) 'fire_type' : 'chemical' or 'non-chemical',

        (if fire) 'average_intensity' : 'none', 'low', 'medium', 'high',
        (if fire) 'fire_type' : 'chemical' or 'non-chemical',

        (if person) 'load' : number corresponding to amt of agents needed,
        (if person) 'status' : 'grabbed' or 'grounded',

        (if reservoir) 'resource_type' : 'sand' or 'water',
        (if reservoir) 'inventory' : dict w/ {resource_type : amt} (likely math.inf),

        (if deposit) 'inventory' : dict w/ {resource_type : amt} for all resource types (including person),

        (if absagent) 'inventory' : dict w/ {resource_type : amt} for all resource types (including person),
        }
    ------------------------------
    """
    MAX_STEPS_FOR_EXPLORE=20
    seed=None # must initialize seed

    def __init__(self):
        pass

    @staticmethod
    def convert_dict_to_string(input_dict : dict, pre : str="") -> str:
        """
        add new lines for each key value pair
        Example output:
        {Task: bring a tomato, lettuce and bread to the countertop to make a sandwich
        Alice's observation: I see: ['Cabinet_1', 'UpperCabinets_1', 'StandardCounterHeightWidth_1', 'Wall_1', 'Ceiling_1', 'Fridge_1', 'Toaster_1', 'CoffeeMachine_1', 'CounterTop_1']
        Alice's state: I am at co-ordinates: (-1.00, 0.90, 1.00) and I am holding nothing
        Bob's observation: I see: ['Cabinet_1', 'Sink_1', 'Mug_1', 'UpperCabinets_1', 'StandardCounterHeightWidth_1', 'Wall_1', 'Window_1', 'Fridge_1', 'Toaster_1', 'CoffeeMachine_1', 'PaperTowelRoll_1', 'CounterTop_1']
        Bob's state: I am at co-ordinates: (-1.00, 0.90, 0.00) and I am holding nothing
        History: {'Observations': {'Alice': [], 'Bob': []}, 'Actions': {'Alice': [], 'Bob': []}, 'Action Success': {'Alice': [], 'Bob': []}} }
        """
        return "{\n" + "\n".join(f"{pre}{k}: {v}, " for k, v in input_dict.items()) + "\n"+f"{pre}}}"

    def get_act_failure_text(self, actions: List[str], agent_idx: int) -> str:
        """Get a text describing that the previous actions failed
        actions: List of failed actions, agent_id: int
        actions = [<action_1>, <action_2>, ..., <action_n>]
        Example: Previously, I have tried to <action_1>, <action_2>, ..., <action_n> but was unsuccessful.
        Previously, I have tried to put the Egg_1 in my hand on CounterTop_1, move ahead, rotate left but was unsuccessful.
        """

        """
        MOVEMENT_ACTIONS=['NavigateTo', 'Move']
        CARRY_DROP_ACTIONS=['Carry', 'DropOff']
        SUPPLY_ACTIONS=['StoreSupply', 'UseSupply', 'GetSupply', 'ClearInventory']
        """
        # TODO: how to add two parameters to action

        if len(actions) == 0:
            return "None"
        start_phrase = "Previously, I have tried to "
        end_phrase = ", but was unsuccessful"

        act_texts=[]
        for action in actions:
            act_text=self.parse_action(action, agent_idx, return_dict=False)
            act_texts.append(act_text)

        # merge
        inner_phrase=','.join(act_texts)
        failure_text=start_phrase+inner_phrase+end_phrase
        return failure_text

    def get_act_text(self, action: str, act_success: bool, agent_idx: int, error_type: str = None) -> str:
        """Get a text describing what changes the action taken in the previous step"""

        start_text="I tried to"
        success_text="and was successful."
        unsuccess_text="and was not successful."
        end_text=success_text if act_success else unsuccess_text
        # use fn below
        act_text=self.parse_action(action, agent_idx, return_dict=False)

        if error_type=="not_visible":
            middle_text=" but it wasn't visible "
        elif error_type=="not_interactable":
            middle_text=" but I wasn't close enough "
        elif error_type=="restricted_action":
            middle_text=" but I was already holding person "
        else:
            middle_text=" "

        s=f"{start_text} {act_text}{middle_text}{end_text}"
        return s

    def parse_action(self, action : str, agent_idx : int, return_dict=True):
        # parse action string (in format specified in LLM prompt) to format usable to controller
        # CAN return list of kwargs (in the case of Explore for instance)
        # RETURN: formatted_action
        
        if action in ["Done", "Idle"]:
            action="NoOp()"

        action_name = action.split("(")[0]
        # NOTE: action_inner could be two (i.e. "GreatFire, Sand")
        action_inner = action.split("(")[1].split(")")[0]

        action_dict={}
        action_dict['agent_idx']=agent_idx
        action_dict['action']=action_name

        # such as NavigateTo -> navigate to GreatFire, etc
        # use in get_act_failure_text fn
        act_text=None

        """
        Formatting for actions, required action_args:

        'NavigateTo' : to_target_id (location),
        'Move' : to_target_id (direction), # Up, Down, Left, Right
        'Carry' : from_target_id (person),
        'DropOff' : from_target_id (person), to_target_id (deposit)
        'StoreSupply' : to_target_id (deposit),
        'UseSupply' : to_target_id (fire), supply_type (on fire)
        'GetSupply' : from_target_id (deposit or reservoir), supply_type (deposit)
        'ClearInventory',
        'NoOp',

        (extra) 'Explore'

        Types of error_type(s):
        -restricted_action, not_visible, not_interactable
        """

        # NOTE: can only use this function when we have get_id (i.e. called from env, has controller)
        # yet we're putting it in BaseEnv to avoid clutter
        assert hasattr(self, 'get_id'), f"Cannot call this function without having .get_id(.)"

        # Movement actions
        if action_name=="NavigateTo":
            action_dict["to_target_id"]=self.get_id(action_inner) # object
            act_text=f"navigate to {action_inner}"
        if action_name=="Move":
            action_dict["to_target_id"]=action_inner # direction
            act_text=f"move {action_inner}"
        if action_name=="Explore":
            # we assume that this is called when we have the controller, so we use the position of the agent
            action_dict=self.explore_actions(agent_idx=agent_idx)
            act_text="explore"

        # Carry Drop actions
        if action_name=="Carry":
            action_dict["from_target_id"]=self.get_id(action_inner) # person
            act_text=f"carry {action_inner}"
        if action_name=="DropOff":
            deposit_name,person_name=action_inner.split(", ")
            action_dict["to_target_id"]=self.get_id(deposit_name) # deposit
            action_dict["from_target_id"]=self.get_id(person_name) # person
            act_text=f"drop off {person_name} at {deposit_name}"

        # supply actions
        if action_name=="StoreSupply":
            action_dict["to_target_id"]=self.get_id(action_inner) # deposit
            act_text=f"store all my supplies at {action_inner}"
        if action_name=="UseSupply":
            fire_name,supply_type=action_inner.split(", ")
            action_dict["to_target_id"]=self.get_id(fire_name) # fire
            action_dict["supply_type"]=supply_type # supply
            act_text=f"use {supply_type.lower()} on {fire_name}"
        if action_name=="GetSupply":
            # NOTE: by convention enforced on procedural generation, the name contains the object type on it (except for agent)
            if "Deposit" in action_inner:
                deposit_name,supply_type=action_inner.split(", ")
                action_dict["from_target_id"]=self.get_id(deposit_name) # deposit
                action_dict["supply_type"]=supply_type # supply
                act_text=f"get {supply_type.lower()} from {deposit_name}"
            elif "Reservoir" in action_inner:
                action_dict["from_target_id"]=self.get_id(action_inner) # reservoir 
                act_text=f"get supply from {action_inner}"
        # no need to do anything for action dict - for both below
        if action_name=="ClearInventory":
            act_text="clear inventory"

        # no op actions
        if action_name=="NoOp":
            act_text="be idle"

        if return_dict:
            return action_dict
        return act_text

    def explore_actions(self, agent_idx):
        # NOTE: This fn is sorta thrown together
        # @here 

        # (a or b) <-> not (nota and notb)
        if not hasattr(self, 'exploration_direction_history'):
            self.dir_history=[(0,0)]

        # defining useful fns
        get_rnd_dir=lambda : (random.randint(0,2)-1, random.randint(0,2)-1) # random sample from [-1,0,1] x [-1,0,1]
        abs_sum=lambda l : sum([abs(v) for v in l])
        negative=lambda t : tuple([-tt for tt in t])
        add=lambda t1, t2 : tuple([tt1+tt2 for tt1,tt2 in zip(t1,t2)])
        mul_scalar=lambda t, s : tuple([tt*s for tt in t])
        manhattan=lambda t1, t2: sum([abs(tt1-tt2) for tt1,tt2 in zip(t1,t2)])
        steps=lambda rnd_dir : SARBaseEnv.MAX_STEPS_FOR_EXPLORE // abs_sum(rnd_dir)
        bound_tuple=lambda t : tuple([misc.clamp(t[0], 0, core.Coordinate.WIDTH), misc.clamp(t[1], 0, core.Coordinate.HEIGHT)])
        agent=self.controller.get('agents', agent_idx)

        def truncated_direction(rnd_dir, minimum_perc):
            delta_pos=mul_scalar(rnd_dir, steps(rnd_dir))

            curr_pos=agent.get_position()
            next_pos_theory=add(curr_pos,delta_pos)
            next_pos_actual=bound_tuple(next_pos_theory)

            distance_theory=manhattan(curr_pos, next_pos_theory)
            distance_actual=manhattan(curr_pos, next_pos_actual)
            if distance_theory==0:  return True
            assert 0<=distance_actual/distance_theory<=1, f"Something weird happened in truncated_direction, ratio is not [0,1]"

            # if ratio falls below minimum percent, it's 'truncated'
            return  (distance_actual/distance_theory) < minimum_perc

        # get 'random' direction
        # not entirely random as we don't allow:
        # previous repetition NOR going in opposite direction as previous
        random_direction=get_rnd_dir()
        previous_dir=self.dir_history[-1]

        # while (rnd_dir == previous_rnd_dir) or (rnd_dir == negative(previous_rnd_dir)), keep randomizing
        # OR more than half of the movement in that direction is truncated (obstacle)
        while (random_direction in [previous_dir, negative(previous_dir), (0,0)]) or truncated_direction(random_direction, .75):
            random_direction=get_rnd_dir()
        self.dir_history.append(random_direction)

        dx,dy=random_direction
        explore_actions=[]
        for clk in range(steps(random_direction)):
            # make all these steps inert

            # move in x
            dx_readable=["Left", "Idle", "Right"][dx+1]
            if dx_readable!="Idle":
                action=f"Move({dx_readable})"
                kwargs=self.parse_action(action, agent_idx=agent_idx)
                kwargs['inert_step']=True
                explore_actions.append(kwargs)
            

            # move in y
            dy_readable=["Up", "Idle", "Down"][dy+1]
            if dy_readable!="Idle":
                action=f"Move({dy_readable})"
                kwargs=self.parse_action(action, agent_idx=agent_idx)
                kwargs['inert_step']=True
                explore_actions.append(kwargs)

        explore_actions[-1]['inert_step']=False # make last one non-inert
        return explore_actions

    def stop(self):
        pass
        
"""
llm interface
feedback

error propagation:
    minimal error feedback, explicitly say environment rules in prompt
        -rules including the amt that is collected from deposits, reservoirs, and dropped into fires
        -(rules about low-level movement) which objects are collidable, and why move would fail
        -grid and bounds - scene boundary (can say 'empty' but that can mean boundary)
        -using appropriate resource for type of fire (sand vs. water) - should we give it fire type? (for now yes)
        -using supply on fire can only fail if wrong type supply used, OR if no flammables were neighbooring or on the agent itself
            -chemical, non-chemical -> sand, water mapping

    -fire having average intensity of None means that it's extinguished
    -since fire is raging on, you might have to collect water multiple times; time is of the essence
        -might make more sense to tame fire before it spreads too much

    -carrying agent:
        -can't do navigation action if grounded
        -inventory is full (can't use AND can't clear)


observation description:

    hardcore certain global variables into the prompt (such as inventory capacity, time from ltom,mtoh)
    inventory:
        -DO: emphasize having person in inventory (and how it's EXCLUSIVE!)
"""

"""
Example of input:

{
Task: ,
Alice's observation:
	Directly around me, I can see:
	{
	Left: ['Flammable of fire (intensity None): GreatFire'],
	Up: ['Flammable of fire (intensity None): GreatFire'],
	Center: ['Flammable of fire (intensity None): GreatFire'],
	Down: ['Flammable of fire (intensity None): GreatFire'],
	Right: ['Flammable of fire (intensity None): GreatFire'],
	},
	Globally, I can see: ['CaldorFire with average intensity of Low of Chemical type', 'CaldorFire_Region_1 with an intensity of None of Chemical type', 'CaldorFire_Region_2 with an intensity of None of Chemical type', 'CaldorFire_Region_3 with an intensity of None of Chemical type', 'GreatFire with average intensity of Low of Non-chemical type', 'GreatFire_Region_1 with an intensity of None of Non-chemical type', 'GreatFire_Region_2 with an intensity of None of Non-chemical type', 'GreatFire_Region_3 with an intensity of None of Non-chemical type', 'ReservoirUtah containing Sand', 'ReservoirYork containing Water', "DepositFacility containing {'Sand': 0, 'Water': 0, 'Person': 0}"],
	Names: ['CaldorFire', 'CaldorFire_Region_1', 'CaldorFire_Region_2', 'CaldorFire_Region_3', 'GreatFire', 'GreatFire_Region_1', 'GreatFire_Region_2', 'GreatFire_Region_3', 'ReservoirUtah', 'ReservoirYork', 'DepositFacility'],
Alice's state: I am at co-ordinates: (22, 18) and I am holding {'Sand': 0, 'Water': 1, 'Person': 0}.,
Alice's previous observation: ['CaldorFire', 'CaldorFire_Region_1', 'CaldorFire_Region_2', 'CaldorFire_Region_3', 'GreatFire', 'GreatFire_Region_1', 'GreatFire_Region_2', 'GreatFire_Region_3', 'ReservoirUtah', 'ReservoirYork', 'DepositFacility'],
Alice's previous action: I tried to use water on GreatFire and was successful.,
Alice's previous failures: Previously, I have tried to use water on GreatFire, but was unsuccessful,
Bob's observation:
	Directly around me, I can see:
	{
	Left: ['Flammable of fire (intensity Low): CaldorFire'],
	Up: ['Flammable of fire (intensity Low): CaldorFire'],
	Center: ['Flammable of fire (intensity None): CaldorFire'],
	Down: ['Flammable of fire (intensity None): CaldorFire'],
	Right: ['Flammable of fire (intensity None): CaldorFire'],
	},
	Globally, I can see: ['CaldorFire with average intensity of Low of Chemical type', 'CaldorFire_Region_1 with an intensity of None of Chemical type', 'CaldorFire_Region_2 with an intensity of None of Chemical type', 'CaldorFire_Region_3 with an intensity of None of Chemical type', 'GreatFire with average intensity of Low of Non-chemical type', 'GreatFire_Region_1 with an intensity of None of Non-chemical type', 'GreatFire_Region_2 with an intensity of None of Non-chemical type', 'GreatFire_Region_3 with an intensity of None of Non-chemical type', 'ReservoirUtah containing Sand', 'ReservoirYork containing Water', "DepositFacility containing {'Sand': 0, 'Water': 0, 'Person': 0}"],
	Names: ['CaldorFire', 'CaldorFire_Region_1', 'CaldorFire_Region_2', 'CaldorFire_Region_3', 'GreatFire', 'GreatFire_Region_1', 'GreatFire_Region_2', 'GreatFire_Region_3', 'ReservoirUtah', 'ReservoirYork', 'DepositFacility'],
Bob's state: I am at co-ordinates: (4, 4) and I am holding {'Sand': 1, 'Water': 0, 'Person': 0}.,
Bob's previous observation: ['CaldorFire', 'CaldorFire_Region_1', 'CaldorFire_Region_2', 'CaldorFire_Region_3', 'GreatFire', 'GreatFire_Region_1', 'GreatFire_Region_2', 'GreatFire_Region_3', 'ReservoirUtah', 'ReservoirYork', 'DepositFacility'],
Bob's previous action: I tried to use sand on CaldorFire and was successful.,
Bob's previous failures: Previously, I have tried to use sand on CaldorFire, but was unsuccessful,
Robots' subtasks: [],
Robots' combined memory: [],
Robots' open subtasks: [],
Robots' completed subtasks: [],
}

"""
