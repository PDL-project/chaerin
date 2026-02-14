import copy
from core import Controller
from Scenes.scene_initializer import BaseSceneInitializer
from misc import Arg

class SceneInitializer(BaseSceneInitializer):
    # 30 x 30, contains one of all w/ n_agents
    # 1 big fire, w/ 3 initially lighted locations

    def __init__(self) -> None:
        self.name='area_900_1_fire_3_lights'
        self.params={
                'grid_size' : (30,30,1),

                'reservoirs' : [
                    Arg(tp='a',name='ReservoirTaj',position=(21,6)),
                    Arg(tp='b', name='ReservoirLibre',position=(22,9))
                    ],
                'deposits' : [
                    Arg(name='DepositFacility', position=(24,14)),
                    ],
                'fires' : [
                    Arg(tp='a', amt_light=3, amt_regions=None, enclosing_grid=(10,10), name='RedFire', position=(11,4)),
                    ],
                'persons' : [
                    Arg(extra_load=0,find_probability=.35, name='LostPersonThomas',position=(6,23)),
                    ],
                'agents' : [
                    Arg(position=(7,9)),
                    Arg(position=(24,3)),
                    Arg(position=(25,4)),
                    Arg(position=(25,16)),
                    Arg(position=(25,14)),
                    Arg(position=(25,25))
                    ]
                }
        self.task_timeout=35
        super().__init__()
