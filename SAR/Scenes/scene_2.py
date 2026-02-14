import copy
from core import Controller
from Scenes.scene_initializer import BaseSceneInitializer
from misc import Arg

class SceneInitializer(BaseSceneInitializer):
    # 30 x 30, contains one of all w/ n_agents
    # one fire has 2 initially lighted locations, the other has 1 initially lighted location - but opposite type than scene 1

    def __init__(self) -> None:
        self.name='area_900_two_fires_3_lights'
        # v1 - avoid having more regions than lights, it just creates confusion & redundancy
        self.params={
                'grid_size' : (30,30,1),

                'reservoirs' : [
                    Arg(tp='a',name='ReservoirOmaha',position=(5,19)),
                    Arg(tp='b', name='ReservoirWhosville',position=(9,18))
                    ],
                'deposits' : [
                    Arg(name='DepositFacility', position=(12,27)),
                    ],
                'fires' : [
                    Arg(tp='a', amt_light=1, amt_regions=None, enclosing_grid=(8,8), name='EnglandFire', position=(2,10)),
                    Arg(tp='b', amt_light=2, amt_regions=None, enclosing_grid=(8,8), name='TownFire', position=(14, 15))
                    ],
                'persons' : [
                    Arg(extra_load=0,find_probability=.35, name='LostPersonJeremy',position=(26,4)),
                    ],
                'agents' : [
                    Arg(position=(17,14)),
                    Arg(position=(9,7)),
                    Arg(position=(3,24)),
                    Arg(position=(2,25)),
                    Arg(position=(25,25)),
                    Arg(position=(16,15))
                    ]
                }
        self.task_timeout=35
        super().__init__()
