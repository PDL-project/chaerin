import copy
from core import Controller
from Scenes.scene_initializer import BaseSceneInitializer
from misc import Arg

class SceneInitializer(BaseSceneInitializer):
    # 30 x 30, contains one of all w/ n_agents
    # 3 fires, each with 1 initially lighted location

    def __init__(self) -> None:
        self.name='area_900_3_fires_3_lights'
        self.params={
                'grid_size' : (30,30,1),

                'reservoirs' : [
                    Arg(tp='a',name='ReservoirTaj',position=(6,20)),
                    Arg(tp='b', name='ReservoirLibre',position=(10,17))
                    ],
                'deposits' : [
                    Arg(name='DepositFacility', position=(13,25)),
                    ],
                'fires' : [
                    Arg(tp='a', amt_light=1, amt_regions=None, enclosing_grid=(5,5), name='EmberFire', position=(2,10)),
                    Arg(tp='b', amt_light=1, amt_regions=None, enclosing_grid=(5,5), name='AgniFire', position=(14, 15)),
                    Arg(tp='b', amt_light=1, amt_regions=None, enclosing_grid=(5,5), name='DowntownFire', position=(23, 20))
                    ],
                'persons' : [
                    Arg(extra_load=0,find_probability=.35, name='LostPersonThomas',position=(25,6)),
                    ],
                'agents' : [
                    Arg(position=(9,7)),
                    Arg(position=(3,24)),
                    Arg(position=(2,25)),
                    Arg(position=(17,25)),
                    Arg(position=(16,26)),
                    Arg(position=(25,25))
                    ]
                }
        self.task_timeout=35
        super().__init__()
