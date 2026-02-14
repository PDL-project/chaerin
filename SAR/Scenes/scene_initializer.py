import copy
from core import Controller, Field
from misc import join_conjunction

class BaseSceneInitializer:
    def __init__(self) -> None:
        self.check_proper()
        self.max_num_agents=len(self.params['agents'])

    def get_task(self):
        map_tp=lambda tp : Field.READABLE_TYPE_MAPPER_FIRE[tp.upper()].lower()
        fire_names=[f"{map_tp(arg.tp)} fire {arg.name}" for arg in self.params['fires']]
        person_names=[arg.name for arg in self.params['persons']]
        # v1 - emphasize that finding that person is final (no more)
        task=f"Using the appropriate resources, extinguish all the fires: {join_conjunction(fire_names, 'and')}. Also, explore to find the following lost people (and no more): {join_conjunction(person_names, 'and')}, carry them, and drop them in a deposit."
        return task

    def check_proper(self):
        """ Check if the initialization is proper """
        # self.params exists
        assert hasattr(self, 'params'), f"Parameters have not been created."
        # self.name exists
        assert hasattr(self, 'name'), f"Name for initialization not provided."
        # self.task_timeout exists
        assert hasattr(self, 'task_timeout'), f"Task timeout not provided."

        # naming convention: Name must contain class type (except agents)
        for poi,l in self.params.items():
            if poi in ['grid_size','agents']: continue
            o_type=poi[:-len('s')].capitalize()
            for i,arg in enumerate(l):
                assert o_type in arg.name, f"Object of type {o_type} in initialization does not contain '{o_type}' in name {arg.name}"

        # check there is at least one resevoir for each fire type
        fire_type_spectrum=set([arg.tp for arg in self.params['fires']])
        reservoir_type_spectrum=set([arg.tp for arg in self.params['reservoirs']])
        assert fire_type_spectrum.issubset(reservoir_type_spectrum), f"Not the fire types ({fire_type_spectrum}) must be represented in the reservoir types ({reservoir_type_spectrum})"

        # check there's at least one of all the types
        keys=['reservoirs', 'deposits', 'fires', 'persons', 'agents']
        for k in keys: assert len(self.params.get(k,[]))>0, f"In params, key {k} has no arguments."

        # check none of them are only called by their class name (that's reserved for generics)
        for k in keys[:-1]: # except for agents, they're named separately
            class_type=k[:-1].capitalize()
            for v in self.params[k]: assert (v.name!=class_type), f"Cannot name object with same name as class type ({class_type}), that name is reserved"

        # grid_size should be len-2 tuple
        grid_size=self.params.get('grid_size',[])
        assert isinstance(grid_size, tuple) and len(grid_size)==3, f"Grid size parameter is not tuple, or is wrong size (must have x,y,z)."
        # should be positive
        assert grid_size[0]>0 and grid_size[1]>0, f"Grid size has non-positive entries."


    def preinit(self, num_agents, agent_names, seed):
        """ Returns initialized controller & parameters used (for information only) """
        
        assert 1<=num_agents<=self.max_num_agents, f"For the {self.name} initialization min: 1 and max: {self.max_num_agents} are supported not {num_agents}"
        pg_params=copy.deepcopy(self.params)
        pg_params['agents']=pg_params['agents'][:num_agents]
        # name agents
        for i,args in enumerate(pg_params.get('agents',[])):
            args.name=agent_names[i]
            assert args.name != 'Agent', f"Cannot name agent 'Agent'"

        # ---- initialize controller ----
        controller=Controller(procedural_generation_parameters=pg_params, seed=seed)


        # ---- additional information added ----
        # This is added as extra information (used in feasable action mapping)
        pg_params['available_object_names']=controller.all_names

        # Make the types readable
        for poi,l in pg_params.items():
            for i,arg in enumerate(l):
                if hasattr(arg, 'tp'):
                    # TODO: a bit wonky to use Field class
                    arg.tp=Field.READABLE_TYPE_MAPPER_RESOURCE[arg.tp.upper()].capitalize()

        return controller, pg_params
