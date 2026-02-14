from Scenes.base_checker import BaseChecker
from collections import Counter

class Checker(BaseChecker):
    def __init__(self, params : dict):
        subtasks,coverage=self.initialize(params)
        super().__init__(
                subtasks,
                coverage
                )

    def initialize(self, params):
        """ Use params from the controller initialization to auto-create subtasks """

        subtasks=[]
        coverage=[]

        # give all objects to base checker class in order to properly account for navigate to
        self.available_object_names=params.pop('available_object_names')

        # Check for ambiguity - raise error if so
        supply_to_reservoir={}
        for arg in params["reservoirs"]:
            if supply_to_reservoir.get(arg.tp,None) is not None:
                print("Cannot yet have >1 supply of the same type.")
                raise NotImplementedError
            supply_to_reservoir[arg.tp]=arg.name

        # ----------------------------------------------------------
        for poi,l in params.items():
            # don't add agents, see about reservoirs below
            if poi in ['reservoirs', 'agents']: continue
            names=list(map(lambda a : a.name, l))
            coverage.extend(names)

        # add only reservoirs that have corresponding action
        necessary_types=list(set([a.tp for a in params["fires"]]))
        coverage.extend([supply_to_reservoir[tp] for tp in necessary_types])
        # avoid double-counting - we didn't, just making sure though
        coverage=list(set(coverage))

        # ---------------------- subtasks ---------------------------
        # TODO (as needed): implement this w/ multiple deposits
        #       we'll need to add generics to the checker function
        #       (perhaps do auto credit assignment for navigateto(deposit) when doing dropoff?
        if len(params['deposits'])>1:
            print("Cannot yet use checker w/ more than 1 deposit (ambiguity).")
            raise NotImplementedError
        deposit_singleton=params['deposits'][0]

        # --- fires ---
        for arg in params["fires"]:
            subtasks.append(f"NavigateTo({arg.name})")
            subtasks.append(f"UseSupply({arg.name}, {arg.tp})")
            # IMPORTANT: Add subtask (to be checked independently at the end) that ensures fire is tamed
            subtasks.append(f"EndFire({arg.name})")

        # --- person ---
        for arg in params["persons"]:
            subtasks.append(f"NavigateTo({arg.name})")

            subtasks.append(f"Carry({arg.name})")
            # generic deposit (doesn't matter)
            subtasks.append(f"DropOff({deposit_singleton.name}, {arg.name})")
            subtasks.append(f"NavigateTo({deposit_singleton.name})")
            # IMPORTANT: Add subtask (to be checked independently during callback) to check if person has been spotted
            subtasks.append(f"Spot({arg.name})")

        # --- deposit ---
        # Nothing. Using deposit is not mandatory.

        # --- reservoir ---
        # loop through fires since we use reservoir in an as-needed basis
        for arg in params["fires"]:
            reservoir_name=supply_to_reservoir[arg.tp]
            subtasks.append(f"GetSupply({reservoir_name})")
            subtasks.append(f"NavigateTo({reservoir_name})")
        # ----------------------------------------------------------

        # avoid repetition
        subtasks=list(set(subtasks))
        # print("Checking subtasks:", subtasks)
        # print("Checking coverage:", coverage)

        return subtasks,coverage

    def callback(self):
        """ To check at the end of each multi-agent step if fire has been extinguished """
        for d in self.event.get('global_obs',None):
            if d['type']=='Fire':
                subtask=f"EndFire({d['name']})"
                average_intensity=d.get('average_intensity', '')
                # fire has subsided & not already completed
                if average_intensity=='None' and (subtask not in self.subtasks_completed):
                    self.subtasks_completed.append(subtask)

            elif d['type']=='Person':
                subtask=f"Spot({d['name']})"
                spotted=d.get('spotted', False)
                if spotted and (subtask not in self.subtasks_completed):
                    self.subtasks_completed.append(subtask)
