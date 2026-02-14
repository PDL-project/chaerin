import copy
from core import Controller
from Scenes.scene_initializer import BaseSceneInitializer
from misc import Arg

class SceneInitializer(BaseSceneInitializer):
    # 30 x 30, contains one of all w/ n_agents
    # 1 fire w/ 1 initially lighted, and 2 persons to be rescued - all previous only have 1

    def __init__(self) -> None:
        self.name='area_900_1_fire_2_lights_2_persons'
        self.params={
                'grid_size' : (30,30,1),

                'reservoirs' : [
                    Arg(tp='a',name='ReservoirUtah',position=(6,16)),
                    Arg(tp='b', name='ReservoirYork',position=(9,19))
                    ],
                'deposits' : [
                    Arg(name='DepositFacility', position=(12,24)),
                    ],
                'fires' : [
                    Arg(tp='b', amt_light=1, amt_regions=None, enclosing_grid=(6,6), name='SussexFire', position=(15,21))
                    ],
                'persons' : [
                    Arg(extra_load=0,find_probability=.3, name='LostPersonJacob',position=(18,4)),
                    Arg(extra_load=0,find_probability=.3, name='LostPersonZoe',position=(24,6)),
                    ],
                'agents' : [
                    Arg(position=(7,10)),
                    Arg(position=(14,18)),
                    Arg(position=(15,15)),
                    Arg(position=(4,24)),
                    Arg(position=(2,24)),
                    Arg(position=(26,26))
                    ]
                }
        self.task_timeout=35
        super().__init__()
