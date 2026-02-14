from enum import Enum
from collections import defaultdict
import math
import copy
import functools, itertools
import warnings
import inspect
import random
import time
import uuid
import numpy as np
from misc import *

class Coordinate:
    """
    Default to 3 by 3 by 1 grid, xy mode, discrete.
    Measures from TOP LEFT
    ALWAYS DISCRETE
    """
    
    WIDTH=3 # x-axis
    HEIGHT=3  # y-axis
    ALTITUDE=1 # z-axis

    ALL_AXES='xyz'
    AXES='xy'
    AXES_LIMITS={'x' : 'WIDTH', 'y' : 'HEIGHT', 'z' : 'ALTITUDE'}

    """ -------------------- static methods -------------------- """
    @staticmethod
    def set_axes_mode(s : str):
        """ e.g. .set_axes_mode('xz'), order matters """
        s=s.lower().strip()
        axes=''.join(list(s))
        Coordinate.AXES=axes

    @staticmethod
    def get_axes_limits():
        """ Get limits (inclusive), e.g. for default it's [0,2]. Subtract 1 for discrete as it's zero-indexed """
        return [getattr(Coordinate, Coordinate.AXES_LIMITS[ax])-1 for ax in Coordinate.AXES]

    """ within bounds methods """
    @staticmethod
    def within_bounds(*args):
        limits=Coordinate.get_axes_limits()
        # assert len(args)<=len(limits), f"Not enough axes provided, need all of {list(Coordinate.AXES)}."
        min_l=min(len(limits), len(args))
        for axv,lim in zip(args[:min_l], limits[:min_l]):
            if not (0<=axv<=lim): return False
        return True

    @staticmethod
    def assert_within_bounds(*args):
        limits=Coordinate.get_axes_limits()
        assert Coordinate.within_bounds(*args), f"Coordinate not within bounds {list(Coordinate.AXES)} -> {limits}"

    """ Distance methods """
    @staticmethod
    def delta(c1, c2):
        """ Difference in coordinates, c1-c2 """
        t1=c1.get()
        t2=c2.get()
        return tuple([(v1-v2) for v1,v2 in zip(t1,t2)])

    @staticmethod
    def euclidean(c1, c2):
        dp=Coordinate.delta(c1,c2)
        return math.sqrt(sum([v**2 for v in dp]))

    @staticmethod
    def manhattan(c1, c2):
        dp=Coordinate.delta(c1,c2)
        return sum([abs(v) for v in dp])

    """ Quick and easy initialization """
    def __init__(self, *args, radius=0):
        """ If args provided, then they're assumed to be in order of Coordinate.AXES """

        args=args+tuple( 0 for _ in range(len(Coordinate.AXES)-len(args)) ) # pad w/ zeros

        self.r=radius
        self.coords={ax : axv for ax,axv in zip(Coordinate.AXES, args)}
        Coordinate.assert_within_bounds(*self.coords.values())

    @property
    def radius(self):
        return int(self.r)

    def __eq__(self, other):
        """ Check that all the attributes are equal """
        return (self.r==other.r) and (self.coords == other.coords)

    """ 
    Important note about setter and getter:
    getter  - returns the position accounting for discretization
    setter  - sets the position w/ discretization

    --- set(*get()) = identity operator ---
    """

    def get(self, fn=int):
        # underlying continuous, discrete approximation
        return tuple(fn(v) for v in self.coords.values())

    def set(self, *args, fn=lambda x:x, suppress_bounds=False):
        # underlying continuous
        if not suppress_bounds:
            Coordinate.assert_within_bounds(*args)
        self.coords={ax : fn(axv) for ax,axv in zip(Coordinate.AXES, args)}

    def change_position(self, *dargs):
        # @here
        args=self.get()
        dargs=dargs+tuple( 0 for _ in range(len(args)-len(dargs)) ) # pad w/ zeros

        self.set(*tuple_add(args, dargs), suppress_bounds=True)

    """
    Direct getter and setter functions that directly access position,
    done so that the engine can access directly w/o discrete approximation for non-continuous envs
    """
    def direct_get(self):
        return self.get(fn=lambda x:x)

    @deprecated(comment=".set(.) is already direct")
    def direct_set(self, *args):
        return self.set(*args, fn=lambda x:x)

    """
    have different set/get radius function since position is engine-defined, and radius is an abstraction
    """
    def set_radius(self, r : float): self.r=r
    def get_radius(self): return self.r

    # change the size of the underlying grid
    @staticmethod
    def set_params(**kwargs):
        for axn, axv in kwargs.items():
            setattr(Coordinate, axn.upper(), axv)

    @staticmethod
    def get_params():
        return {axn : getattr(Coordinate, axn) for axn in Coordinate.AXES_LIMITS.values()}
    

    """ inter-coordinate methods """
    def within_radius(self, c):
        """ True if coordinate c is within radius of current object (as defined) """
        # if same object, false (exclude self-referential behavior)
        if id(self)==id(c):
            return False
        d=Coordinate.euclidean(self, c)
        """ Distance is measured continuously (to remove confusion) """
        return (d <= self.radius)

    @staticmethod
    def neighboors(c1, c2, diagonal=False):
        """ Returns true if neighboors (in cardinal directions only, unless diagonal=True), if c1==c2 it's trivially true """
        ptpl=c1.get()

        # if same object, false (exclude self-referential behavior)
        if id(c1)==id(c2):
            return False

        for h in [-1,0,1]:
            for w in [-1,0,1]:
                if not diagonal and abs(h)+abs(w)>1:
                    continue
                if tuple_add(ptpl, (w,h))==c2.get():
                    return True
        return False

    @staticmethod
    def overlap(c1, c2):
        """ Overlap as in ontop, not overlapping radii """
        return c1.get() == c2.get()

"""
IMPORTANT!
Global positioning system.
Whenever a position is set or changed, this class' static variables
position tracking is changed.

This is done for ALL objects with a specified position and an id attribute.
Useful for tracking changes in the world.
"""
class GPS:
    tracker=defaultdict(list)
    id_mapping=defaultdict(None)

    @staticmethod
    def update_track(o, previous_position, future_position):
        oid=o.id

        # update position tracking
        prevc,futurec=previous_position.get(),future_position.get()
        if oid in GPS.tracker[prevc]:
            # make sure not duplicted ONLY if positions are different
            assert (prevc==futurec) or (oid not in GPS.tracker[futurec]), f"Object w/ id {oid} in two positions at once {previous_position.get()} and {future_position.get()}"
            # remove from previous position
            GPS.tracker[prevc].remove(oid)

        # add to new position
        GPS.tracker[futurec].append(oid)

        # add to id_mapping if not already there (so we always have global mapping keeping track)
        GPS.id_mapping[o.id]=o

    @staticmethod
    def at(position):
        # get list of object ids at position
        return GPS.tracker[position.get()]

    @staticmethod
    def near(central_position, radius):
        # get a *smaller* dict of position -> oids
        # for positions within radius of central_position
        # including central position itself
        # measured in euclidean distance

        assert radius>=0, f"No negative radii accepted"
        within_positions=[]

        # search rectangle from central_position w/ width & height -> radius
        half_l=math.ceil(radius/2)
        ctpl=central_position.get()
        for dx in range(-half_l, half_l+1):
            for dy in range(-half_l, half_l+1):
                ntpl=tuple_add(ctpl, (dx,dy))
                if Coordinate.within_bounds(*ntpl):
                    ncoord=Coordinate(*ntpl)
                    within_radius=( Coordinate.euclidean(ncoord, central_position) <= radius )
                    if within_radius: within_positions.append(ncoord)

        # sort positions by distance from center
        within_positions.sort(key=lambda ncoord : Coordinate.euclidean(ncoord, central_position))

        d=dict([(p.get(),GPS.at(p)) for p in within_positions])
        return d

    @staticmethod
    def around(central_position, layer, inclusive, diagonal, get_out_of_bounds=False):
        # get a *smaller* dict of position -> oids
        # for positions at the {layer}th layer of central_position
        # if inclusive then include layers<0

        # must be discrete

        assert layer>=0, f"No negative layers accepted"
        assert isinstance(layer, int), f"Layer must be int"

        def _in_layer(x,y,l):
            if l==0:
                # TODO: change to false if not inclusive?
                return (x==0) and (y==0) and inclusive

            diagonal_bool=(abs(x)!=abs(y)) if not diagonal else True
            inside_layer=(x<=l and y<=l) and diagonal_bool
            if inclusive:
                return inside_layer
            return inside_layer and not (x<=l-1 and x<=l-1)

        within_positions=[]
        out_of_bounds=[]

        # search rectangle from central_position w/ width & height -> radius
        ctpl=central_position.get()
        for dx in range(-layer,layer+1):
            for dy in range(-layer,layer+1):
                ntpl=tuple_add(ctpl, (dx,dy))
                if Coordinate.within_bounds(*ntpl):
                    if _in_layer(dx,dy,layer):
                        ncoord=Coordinate(*ntpl)
                        within_positions.append(ncoord)
                else: out_of_bounds.append(ntpl)

        d=dict([(p.get(),GPS.at(p)) for p in within_positions])
        if not get_out_of_bounds:
            return d
        return d, out_of_bounds


# named decorator, extend class to also possess name (id)
# setattr(K, func.__name__, func)
def named(cls):
    class NameWrapper(cls):
        def __init__(self, *args, **kwargs):
            try:
                self.name=kwargs.pop('name')
            except KeyError:
                self.name=None
            super().__init__(*args, **kwargs)
        def set_name(self, name : str):
            self.name=name
        def get_name(self):
            return self.name
    return NameWrapper

# with id decorator, add unique id to each object (to be used by controller)
def with_id(cls):
    class IdWrapper(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def class_name(self):
            fltr=lambda s : all([ts not in s.lower().strip() for ts in ['object', 'wrapper']])
            base_names=[b.__name__ for b in self.__class__.mro()]
            # NOTE: This filter assumes that all the wrappers have the word "wrapper" in them! (otherwise we can't distinguish the levels of base classes)
            base_names=list(filter(fltr, base_names))
            bn=base_names[0] # assume it's the only one left
            return bn

        @functools.cached_property
        def id(self):
            """ id for object, of the form name {class-type}|{id_gibberish}. Works regardless of whether it has been named"""
            bn=self.class_name()
            unique_id=uuid.uuid4()
            # the id below might not be unique!
            # unique_id=id(self)
            _id=f"{bn}|{unique_id}"
            return _id

    return IdWrapper 

# collidable wrapper, specifies if class is collidable or not
# used for navigation - if collidable then agent can walk on it
def collidable(is_collidable):
    def _collidable(cls):
        class CollidableWrapper(cls):
            def __init__(self, *args, **kwargs):

                super().__init__(*args, **kwargs)
                setattr(cls, 'collidable', is_collidable)

        return CollidableWrapper
    return _collidable

# position decorator, extend behavior of class with position information
# add argument for whether this position should be considered mutable or not
# (mutable is only a formality to provide information, immutability isn't enforced)

# Technical note, we need to have layered decorators to add argument,
# this decorator "factory" returns a decorator that uses the argument without needing to have it passed to it
# Even if there is a default argument, you have to call the decorator "factory" (with_position())
# in order to create the decorator that is used by @

def with_position(mutable):
    def _with_position(cls):
        def generic_sees(self, othr):
            return self.position.within_radius(othr.position)

        def person_sees(self, othr, strict=False):
            """ Hide in plain sight
                -within radius (like parent class method)
                -AND not deposited (if it's deposited, it shouldn't be visible or interactable in any way)

                NOTE: Very special feature of person - 
                if any agent sees this person, then its radius will be "infinite" (size of map)
            """
            is_agent=(othr.class_name()=="AbsAgent")
            can_see=self.position.within_radius(othr.position)
            # if agent sees, it has been spotted -- and is thus visible to all other agents
            if is_agent and can_see:
                self.spotted=True

            # all negated if deposited (should be invisible)
            bool_condition=can_see if strict else (can_see or self.spotted)
            return bool_condition and (not self.deposited)

        class PositionWrapper(cls):
            def __init__(self, *args, **kwargs):

                # initialize position default to (0,0)
                try:
                    xx,yy=kwargs.pop('position')
                    self.position = Coordinate(xx,yy)
                except KeyError:
                    self.position = Coordinate(0,0)
                # make sure at initialization the positions of objects are tracked
                self._update_track(future_coords=self.position.get())

                try:
                    r=kwargs.pop('radius')
                    self.position.set_radius(r)
                except KeyError:
                    # leave default to 0
                    pass

                super().__init__(*args, **kwargs)

                # add different .sees(.) class here depending on which class it is
                # do this for hiding objects in plain sight
                # e.g. Person after it has been dropped
                base_names=[b.__name__ for b in cls.mro()]
                if 'Person' in base_names:
                    setattr(cls, 'sees', person_sees)
                else:
                    setattr(cls, 'sees', generic_sees)

            # ------ set/get positions ----

            def _update_track(self, future_coords):
                future_position=Coordinate(*future_coords)
                if hasattr(self, 'id'):
                    GPS.update_track(o=self, previous_position=self.position, future_position=future_position)

            def set_position(self, *args):
                self._update_track(args)
                self.position.set(*args)

            def set_position_direct(self, *args):
                self._update_track(args)
                self.position.direct_set(*args)

            def get_position(self):
                return self.position.get()

            def get_position_direct(self):
                return self.position.direct_get()

            # ---------------------------

            def delta(self, othr):
                """ gives othr-self for position """
                return Coordinate.delta(othr.position, self.position)

            def neighboors(self, othr, diagonal=True):
                return Coordinate.neighboors(self.position, othr.position, diagonal=diagonal)

            def change_position(self,dx,dy):
                self.position.change_position(dx,dy)

            def set_radius(self, r : float):
                self.position.set_radius(r)

            def get_radius(self):
                return self.position.get_radius()

            @functools.cached_property
            def mutable_position(self):
                return mutable


        return PositionWrapper
    return _with_position

"""
Flammable class

-type : A,B
    -A requires water
    -B requires sand

    *flammable material can be made non-flammable (none -> low impossible) if using "pre-extinguish"
    *resources are deposited at a point with a radius of effect
    *fires won't stop until put out (so #steps & #on-fire should be part of reward function)

parameters:

discrete version
-intensity : none (no fire here), low, medium, high
-steps until change intensity: 4 (l to m), 3 (m to h)
-critical intensity to spread : medium

*using proper resource
-extinguishing resource - 1 bucket = 1 level of intensity knocked down, knock down to none to extinguish
    -have splash zone

fire spread:
    -NOTE: fire will remain at constant intensity w/ no spreading from source until encountered by at least one agent
    -fire can only spread to other flammable material if it is a direct neighboor
    

clock - keeps timer
onpause - if onpause then the .step() is impotent (onpause until source is discovered by the agent?)

.light() from none to low
    -done @ initalization
    -done by other flammable class
    
.step()
    -carries out the appropriate updates in intensity (unless onpause)
    -if applicable .spread()
    -update clock

.spread()
    -spreads to other fire objects (how to implement, one big class?)
"""

Intensity = Enum('Intensity', ['NONE', 'LOW', 'MEDIUM', 'HIGH'])

@collidable(is_collidable=False)
@with_id
@with_position(mutable=False)
@named
class Flammable:
    # TODO: tweak these
    L_TO_M=3
    M_TO_H=3
    CRITICAL_INTENSITY=Intensity.MEDIUM
    TYPES=["A", "B"]

    def __init__(self, fire_type : str, intensity : Intensity = Intensity.NONE):
        self.fire_type=fire_type.upper().strip()
        assert self.fire_type in Flammable.TYPES, f"Extinguisher type for fire must be of one of the following types {Flammable.TYPES}"

        # initialize intensity of the fire (NONE -> no fire)
        self.intensity=intensity
        # keep internal clock to change states (changed only from above, by .step() actions)
        self._clock = 0

    def on_fire(self):
        return (self.intensity.value)>1

    def spreadable(self):
        """ Test if fire is spreadable (once it reaches critical intensity) """
        return self.intensity.value >= Flammable.CRITICAL_INTENSITY.value

    def extinguish(self):
        self.intensity=Intensity.NONE
        return True

    def lessen(self, extinguisher_type : str):
        """ returns success or failure; failure if wrong extinguisher_type is used """

        extinguisher_type=extinguisher_type.upper().strip()
        assert extinguisher_type in Flammable.TYPES, f"Extinguisher type for fire must be of one of the following types {Flammable.TYPES}"

        # lower intensity by a notch (when using proper extinguisher)
        if self.fire_type == extinguisher_type:
            self.intensity=Intensity(
                    clamp(self.intensity.value-1, 1, 4)
                    )
            # go back to that intensity's clock
            self.clock_back()
            return True
        return False

    def clock_back(self):
        # NOTE: If an agent lessens a fire and it remains above the critical range, then in the next-step
        # it will regain it's strength back to exactly the same point.
        # Thus, we push clock back to counteract this effect
        # depending on the intensity, we go back to that intensity's beginning clock

        if self.intensity in [Intensity.LOW, Intensity.MEDIUM]:
            self._clock=0   # to beginning
        elif self.intensity == Intensity.HIGH:
            self._clock=Flammable.L_TO_M    # to start of being a medium

        # self._clock=clamp(self._clock-1, 0, math.inf)
        # self._clock-=1 # can be negative, in the next .step() it'll be back to the position we want it

    def light(self, intensity=Intensity.LOW):
        """ Light on fire if it isn't already. Default is low """
        if not self.on_fire():
            self.intensity=intensity
            assert self.on_fire(), f"Flammable .light()-ed, but no .on_fire()"
            return True
        return False

    def step(self):
        # change intensity
        # fire will remain high forever if high

        if self.intensity == Intensity.LOW:
            if self._clock == Flammable.L_TO_M:
                self.intensity = Intensity.MEDIUM

        if self.intensity == Intensity.MEDIUM:
            if self._clock == Flammable.L_TO_M + Flammable.M_TO_H:
                self.intensity = Intensity.HIGH

        # IMPORTANT: only update clock if on fire
        if self.on_fire():
            self._clock+=1
            return True
        return False

@with_id
@named
@with_position(mutable=False)
@collidable(is_collidable=False)
class Fire:
    
    STEP_EVERY=2

    def __init__(self, flammables=[], impotent=True):
        """
        Assuming that flammables are the wrapped version with relative positions

        If fire's global clock is less than onpause, then .step() should be impotent (do nothing)
        this is to add functionality to let agent find the fire before it rapidly spreads
        yet not long enough to have agent collect all resources beforehand

        NOTE: Fires are initially impotent by default
        """

        # assert fire is of the same type for non-zero sets
        if len(flammables)>0:
            tp=flammables[0].fire_type
            for fl in flammables:
                assert fl.fire_type==tp, f"Fire types of flammables in Fire must be equal."

        self.flammables = flammables

        # (for flammables) mapping from id to object for easy search later on; static since id doesn't change
        self.id_mapping=dict([(o.id, o) for o in self.flammables])
        
        # This is all below made faster - now x10 the speed on test and linear instead of quadratic

        self.dneighboor = defaultdict(list)
        for i,fl in enumerate(self.flammables):
            self._add_flammable_neighboors(fl)
            # can be none if name provided after initialization
            fl.parent_name=self.get_name() # tell its fire parent's name

        self.impotent=True
        self._clock=0

    def steppable(self):
        return (self._clock % Fire.STEP_EVERY == 0)

    @functools.cache
    def id_get(self, _id):
        o=self.id_mapping.get(_id, None)
        return o

    def make_impotent(self):
        self.impotent=True

    def make_potent(self):
        self.impotent=False

    @functools.cached_property
    def fire_type(self):
        if len(self.flammables)>0:
            return self.flammables[0].fire_type
        return None

    @property
    def average_intensity(self):
        if len(self.flammables)>0:
            intensities=[f.intensity.value for f in self.flammables]
            avg=round(sum(intensities) / len(intensities))

            # make sure that if there is ANY on_fire() objects, it will have average at least low
            # assuming Intensity(1) is None
            # subtract len(.) since it's 1-indexed
            if sum(intensities)-len(intensities)>0: avg=clamp(avg, 2, avg)
            # otherwise, don't clamp
            return Intensity(avg)
        return None

    def average_neighboor_intensity(self, fl : Flammable):
        """ Gives average intensity of immediate neighboors """
        # NOTE: changed to maximum! (since average can be too misleading about the problem's source)

        intensities=[f.intensity.value for f in self.dneighboor[fl.id]]+[fl.intensity.value] # include itself
        maximum=max(intensities)
        return Intensity(maximum)

        """
        if len(intensities)==0: return None
        avg=round(sum(intensities) / len(intensities))
        # make sure that if there is ANY on_fire() objects, it will have average at least low
        # assuming Intensity(1) is None
        # subtract len(.) since it's 1-indexed
        if sum(intensities)-len(intensities)>0: avg=clamp(avg, 2, avg)
        return Intensity(avg)
        """

    def _add_flammable_neighboors(self, fl : Flammable):
        dn=GPS.around(central_position=fl.position, layer=1, inclusive=True, diagonal=True)
        # gives dn w/ position key and oid list value
        if len(dn.values())>0: neighboor_ids=functools.reduce(lambda l1,l2 : l1+l2, list(dn.values()))
        else: neighboor_ids=[]
        raw_neighboors=[self.id_mapping.get(nid,None) for nid in neighboor_ids]

        # remove None (not applicable to this instance)
        neighboors=list(filter(lambda o: o is not None, raw_neighboors))

        # remove if same position
        neighboors=list(filter(lambda o: o.get_position()!=fl.get_position(), neighboors)) # don't include self in neighboor
        self.dneighboor[fl.id]=neighboors
    
    def add_flammable(self, fl : Flammable):
        self.flammables.append(fl)
        # tell it it's fire parent
        fl.parent_name=self.name
        
        # add to id dictionary
        self.id_mapping[fl.id]=fl

        for fl in self.flammables:
            # update dneighboor
            self._add_flammable_neighboors(fl)

        return True

    def all_objects(self, expand=True, with_memory=False):
        """ returns all constituent objects NOT including itself """
        return self.flammables

    def spread(self):
        """ implement spreading for fire with all flammable objects together """
        
        for i,fl in enumerate(self.flammables):
            # spread to all neighboors if flammable spreadable()
            if fl.spreadable():
                for nfl in self.dneighboor[fl.id]:
                    # spread -> light on fire (if not)
                    success=nfl.light()

    def step(self):
        """ step for entire fire object (would act as conglomerate) (spread and THEN step) """

        # no updating (keep null) if impotent
        # TODO: keep tabs on this, @here
        # AND must be steppable (step every nth)
        if not self.impotent:
            self._clock+=1 # always update clock
            if self.steppable():
                self.spread()
                # True or False at .step() is only an indication of impotence (fire can't fail per-se)
                for fl in self.flammables:
                    clk_before=fl._clock
                    success=fl.step()
                    clk_after=fl._clock
                    assert (clk_after-clk_before)==int(success), f"Successfully stepped but clock didn't change"
            return True
        return False

    def lessen(self, loc, extinguisher_type, diagonal=True):
        """ 
        lessen fire (drop) at location and neighboors of location, w/ extinguisher_type
        location as Coordinate object
        if diagonal is true, then do rectangular (rather than cross) splash region

        returns successes list with flammable object,success pairs 

        """
        #NOTE: subtle error can happen if loc is the same .position object as one of the flammables,
        #    in that case, Coordinate.neighboors will be false (due to the implt. of the fn)
        #    Thus, we do a deepcopy
        loc=copy.deepcopy(loc)

        successes=[]
        for fl in self.flammables:
            if Coordinate.neighboors(fl.position, loc, diagonal=diagonal):
                success=fl.lessen(extinguisher_type)
                successes.append(success)
        any_success=any(successes)

        return any_success

    @staticmethod
    def procedural_generation(fire_type, amt_light, max_w, max_h, proportion_filled, top_left, amt_regions=None, fire_name=None, seed=42, shape='circle'):
        """
        Procedural generation class for fire given size constraints and proportion to be filled
        -fire_type
        -amt_light - amt of flammables to .light()
        -max_w - max width of bouding area
        -max_h - max height of bouding area
        -proportion_filled - proportion of this area to fill
        -top_left - position of top_left corner
        -fire_name - name of fire

        The exact details of the distribution is randomized (within constraints of shape)
        generate names for individual flammable POIs within the fire

        Possible shapes of interest to generate:
            factors: growth of spread, geometry of spread, pinch points
            -random (fire spreads randomly, MUST be connected)
            -triangle (fire spreads in 1 dimension) 
            -circle (fire spreads radially)
            -lines (fire spreads across)

        Calibration for radius size for each POI 
            -radius must not exceed bounding box
        Creation of fire shape & pinch-points
        Make sure that all reservoirs AND deposits have infinite capacity (see rationale in controller class comments)
        Proper scaling to multiple agents (amount of deposits, reservoirs, etc)
        """

        set_seed(seed)
        area=max_w * max_h
        x_bnd,y_bnd=max_w,max_h
        xc,yc=int(x_bnd/2),int(y_bnd/2)
        dx,dy=top_left
        fire_radius=None

        assert amt_light>=1, f"Cannot have no .light in a fire"

        # ------ generate the geometry (put into np array) ------
        if shape=='circle':
            # -------- get radius to fill proportion ------
            if proportion_filled < math.pi/4:
                # there exists an inverse function
                r=math.sqrt(proportion_filled * area / math.pi)
            else:
                # inverse function - approximate radius -> proportion
                # (radius,prop). (min(w,h),pi/4) to (1/2*sqrt(w^2+h^2),1)

                rise=(1-math.pi/4)
                run=(math.sqrt(max_w**2+max_h**2) - 1/2*min(max_w,max_h))
                b=math.pi/4
                xi=1/2*min(max_w,max_h)
                exp=4

                alpha=(1-b)/(run)**(1/exp)
                inv=lambda x : ((x-b)/alpha)**exp+xi

                r=inv(proportion_filled)

            # ------ make flammable centers ------
            # evenly distribute around smaller circle w/ radius (2/3)*r
            rr=(2/3)*r
            rr=clamp(rr, 0, 1/2*min(max_w,max_h) * 3/4)
            init=random.random()*2*math.pi  # randomness

            angles=[(i*(2*math.pi / (amt_light))+init)%(2*math.pi) for i in range(amt_light)] # for light
            # angles=[(i*(2*math.pi / (amt_regions))+init)%(2*math.pi) for i in range(amt_regions)] # both for light & extra regions

            light_points=[( int(math.floor(rr)*math.cos(theta))+xc, int(math.floor(rr)*math.sin(theta))+yc) for theta in angles]
            random.shuffle(light_points) # randomness

            for p1 in light_points:
                for p2 in light_points:
                    if id(p1)==id(p2):
                        continue
                    assert p1!=p2, 'Bounding box or area is too small to fit all the initial .light() positions evenly'


            # ------ create flammables -------
            light_point_cnt=0
            flammables=[]
            for y in range(y_bnd):
                for x in range(x_bnd):
                    in_circle=math.sqrt((x-xc)**2 + (y-yc)**2)<r # 1 if true, 0 otherwise

                    if in_circle:
                        ps=(x+dx,y+dy)
                        fl=Flammable(fire_type=fire_type, position=ps)
                        # set 1.5-layer neighboors radius
                        fl.set_radius(1.5*math.sqrt(2))

                        if (x,y) in light_points:

                            # light only amt_light amt
                            if light_point_cnt<amt_light: fl.light()

                            # name the flames if name provided and that's the mode
                            if Controller.FIRE_SUBDIVISION and (fire_name is not None):
                                fl.set_name(f"{fire_name}_Region_{light_point_cnt+1}")
                                light_point_cnt+=1

                        flammables.append(fl)

            # set the radius of the fire object created
            fire_radius=math.ceil(r)

        else:
            raise NotImplementedError


        # set position to one of the regions (a lighted one, if such exists).
        x,y=light_points[-1]
        fire_position=(x+dx, y+dy)
        fire=Fire(flammables=flammables, position=fire_position, name=fire_name)

        # ------ once fire is made, create the extra regions for easy access ------
        # Do after creation of fire since we'll know all of our neighboors

        assert amt_regions is None, f"Giving specific amount of regions is a deprecated feature"
        # add amt_regions until all places are covered!
        def get_reached_flammables():
            reached=set()
            for fl in flammables:
                # if it has a name (namely, it's a region), then add all direct neighboors
                if fl.get_name() is not None:
                    reached.add(fl)
                    for nfl in fire.dneighboor[fl.id]:
                        reached.add(nfl)
            assert len(reached)<=len(flammables), f"Cannot have more reached than available flammables"
            return reached

        get_unreached_flammables=lambda reached : set(flammables)-reached

        reached_flammables=get_reached_flammables()
        unreached_flammables=get_unreached_flammables(reached_flammables)

        while len(unreached_flammables)>0:
            # randomly sample
            fl=random.choice(list(unreached_flammables))

            # add to regions by giving it a name
            fl.set_name(f"{fire_name}_Region_{light_point_cnt+1}")
            light_point_cnt+=1
            
            # update sets
            reached_flammables=get_reached_flammables()
            unreached_flammables=get_unreached_flammables(reached_flammables)

        # print("Needed to create:", light_point_cnt+1)

        # ----- set radius of fire ----------
        fire.set_radius(fire_radius)

        return fire

@with_id
@named
@with_position(mutable=False)
@collidable(is_collidable=True)
class Reservoir:
    TYPES=['A', 'B']

    def __init__(self, resource_type, available=None):
        # infinite resource available if limit not provided
        if available is None:
            available=math.inf

        assert resource_type.upper() in Reservoir.TYPES, f"Resource type must be of the following types only: {Reservoir.TYPES}"
        self.type=resource_type.upper()
        self.left=available

    def set_available(self, available : int):
        assert available>=0, "Can't set available to negative number."
        self.left=available

    @property
    def available(self):
        return self.left

    @property
    def empty(self):
        return self.left<=0
    
    def use(self, amt):
        used=min(amt, self.left)
        self.left-=used
        return used
    
"""
Deposit class

-storage info : dict w/ type as key and amt as value
-useful get and set functions

NOTE: has position, but radius in which this object would be visible (so shape is effectively circular)
"""

@with_id
@named
@with_position(mutable=False)
@collidable(is_collidable=True)
class Deposit:
    # resource depositing
    TYPES=['A', 'B', 'PERSON']
    PERSON=TYPES[-1]

    def __init__(self, capacity=None, storage=None):
        if capacity is None:
            capacity=math.inf  # infinite capacity unless specified
        self.capacity=capacity

        if storage is None:
            # storage=defaultdict(int)
            storage=dict([(k,0) for k in Deposit.TYPES])
        else:
            upper_keys(storage)
            assert set(Deposit.TYPES)==set(storage.keys()), f"Storage dict (resource) provided @ init must have all the keys: {Deposit.TYPES}"

        self.storage=storage
        assert self.space<=capacity, f"Storage not within capacity {capacity}"

    def _assert_type(self, rtype):
        assert rtype in Deposit.TYPES, f"Resource type must be of one of these types: {Deposit.TYPES}"
    
    @property
    def space(self):
        return self.capacity-sum(self.storage.values())

    @property
    def full(self):
        return self.space==0

    def available(self, rtype : str):
        rtype=rtype.upper()
        self._assert_type(rtype)
        return self.storage[rtype]
    
    def use(self, rtype : str, amt : int):
        rtype=rtype.upper()
        assert (rtype!=Deposit.PERSON), '.use(.) undefined for person (Deposit class)'
        self._assert_type(rtype)

        used=min(amt, self.available(rtype))
        self.storage[rtype]-=used
        return used

    def store(self, rtype : str, amt : int):
        rtype=rtype.upper()
        self._assert_type(rtype)

        stored=min(amt, self.space)
        self.storage[rtype]+=stored
        return stored

    def store_all(self, storage : dict):
        upper_keys(storage)
        assert set(Deposit.TYPES)==set(storage.keys()), f"Storage dict (resource) provided to store_all(.) must have all the keys: {Deposit.TYPES}"
        amt_storage=sum(storage.values())
        enough_space=(self.space>=amt_storage)
        if not enough_space:
            return False

        for k in self.storage.keys():
            self.store(k, storage[k])
        return True
    
"""
People(group) class

class of stranded people

note: regardless, need 2 to carry 1 person
-amt of people
-(derived) necessary agents to pick up

functions:
    -pickUp given number of agents
"""
PersonStatus = Enum('PersonStatus', ['GRABBED', 'GROUNDED'])
""" Has radius (radius defines its visibility) """

@with_id
@named
@with_position(mutable=True)
@collidable(is_collidable=True)
class Person:
    # minimium amount of agents required to pick up
    MIN_REQUIRED_AGENTS=2

    def __init__(self, extra_load=0):
        # additional load from person additionally from MIN_REQUIRED_AGENTS
        assert extra_load>=0, f"Extra load cannot be negative"
        self.extraload=extra_load

        # object that the person is coupled to (can be from agent for example)
        # must have .position variable
        self.coupled={}
        # uncoupled contains list of agents who WISH to uncouple/drop (once this reaches critical capacity i.e. >=load, then coupled and uncoupled is cleared)
        self.uncoupled={}

        # to check if person is finished with trajectory (it has been deposited)
        self.deposited=False
        # check if it has been spotted (if it has, then it's now visible to all agents)
        self.spotted=False

        self._clock=0

    @functools.cached_property
    def load(self):
        return self.extraload+Person.MIN_REQUIRED_AGENTS

    @property
    def exceeded_load(self):
        return len(self.coupled.keys())>=self.load

    @property
    def status(self):
        # update status to proper one
        if self.exceeded_load: return PersonStatus.GRABBED
        else: return PersonStatus.GROUNDED

    def depositable(self, deposit : Deposit):
        """
        Only checks if depositable @ location NOT if it's droppable overall (since it doesn't check if it's grabbed in the first place)
        -all agents must be within radius
        -there must be space in deposit
        """
        s=[]
        for agent_id, agent in self.coupled.items():
            close=deposit.sees(agent)
            s.append(close)

        # NOTE: By the implementation, if at least one agent isn't close enough, then it can't be dropped
        #       EVEN IF that extra agent is redudant (i.e. the close ones have a critical mass)
        #       This is to keep real-worldness assumptions & punish redundancy
        enough_space=not deposit.full
        return all(s) and enough_space

    @property
    def grabbed(self):
        return (self.status==PersonStatus.GRABBED)

    def pick(self, agent):
        """
        Add agent to coupled, so it can be picked up
        Success if person is within radius
        """
        info={
            'visible' : None,
            'previously_picked' : None,
            }
        
        # only make this happen if within radius of person
        # additionally, only make this happen if not already grabbed (sufficient agents)
        # visible -> within radius in the pick scenario
        visible=self.sees(agent, strict=True)

        info['visible']=visible

        if visible and (not self.grabbed):
            # gives self parameter to agent for memory
            # not successful if agent already has person picked up
            success=agent._add_person(self)
            # upd. info
            info['previously_picked']=success
            if success:
                self.coupled[agent.id]=agent
                return True,info

        return False,info

    def drop(self, agent, deposit : Deposit):
        """ 
        if False (failure), then agent wasn't carrying it in the first place or drop(.) from this agent wasn't enough to drop person
        NOTE: by the implementation of this function, only successful if all coupled agents drop it (so if another agent is coupled mid-way, that counts)
              thus, when giving success feedback to LLM, say success for all drop actions if ANY one of them is successful
        """
        # NOTE: LLM: 
        #   -give information of failures (maybe? our system should perform without oracle feedback anyways...)
        #       -when there isn't enough to pick up (critical mass information)
        #       -can't drop because it's not even grabbed
        #       -not enough critical mass of dropped
        #   -tell in prompt that agents can only rid themselves/drop person if they ALL do it
        #       otherwise they will still be restricted in actions

        # LIST OF ALL CHECKED CONDITIONS (DROP IF:)
        # local (drop this one): coupled beforehand
        # global (drop overall):
        #   -depositable (close enough to drop location & enough space)
        #   -grabbed (previously picked up by critical mass of agents)
            
        info={
            'grabbed' : None,
            'depositable' : None,
            'all_dropped' : None,
            'interactable' : None,
            }
        # add information about whether it's interactable *raw version*
        info['interactable']=deposit.sees(agent)

        # can only add to uncoupled list if it has been coupled beforehand
        local_success=agent.id in self.coupled.keys()

        # not locally successful if agent hasn't picked person up
        if local_success:
            # global success is if the agent has finally been dropped (not same thing as being grounded as that doesn't mean it's in the deposit place)
            global_success,_info=self._drop(agent, deposit)
            for k,v in _info.items():   info[k]=v
            return global_success,info
        return False,info

    def _drop(self, agent, deposit : Deposit):
        """
        Drops if ALL coupled are uncoupled (not only critical mass).
        Notice that dropping is not trivial, it must be done by all agents
        (you can't just pick up and drop, agent is stuck there)

        This method will clear both coupled and uncoupled

        NOTE: If .drop(.) is done twice by same agent, we don't double count this.
               We .drop(.), then it'll add to uncoupled ONLY IF it's close enough
        """

        # FIRST: make sure that if it's close enough, you do add to uncoupled list
        #       Otherwise (if uncoupled added later), the all_dropped bool won't be counted as successful
        depositable=self.depositable(deposit)
        if depositable:
            self.uncoupled[agent.id]=agent

        amt_uncoupled=len(self.uncoupled.keys())
        amt_coupled=len(self.coupled.keys())
        all_dropped=(amt_coupled == amt_uncoupled)

        grabbed=self.grabbed
        if grabbed: assert amt_coupled>=self.load, f"Something has gone wrong, object is grabbed, yet amount coupled isn't higher or equal to the load"

        # CONDITION FOR CHECKING IF DROPPABLE - drop if currently grabbed & depositable & all the agents have dropped it
        droppable=grabbed and depositable and all_dropped
        info={
            'grabbed' : grabbed,
            'depositable' : depositable,
            'all_dropped' : all_dropped,
            }

        if droppable:
            # clear out both coupled and uncoupled
            # @bug, forgot about removing
            for _id, agent in self.coupled.items():
                agent._remove_person()

            self.coupled={}
            self.uncoupled={}

            # Note that deposit does NOT need to know person instance deposited, assume that it's black hole (person will stay there forever)
            # put in the deposit location
            used=deposit.store(Deposit.PERSON, 1)
            assert used>0, f"Something has gone wrong, depositable was true yet there was not space for deposit (deposit object was likely changed in-between)"

            # set the position to position of deposit
            # otherwise, it'll be in an agent position and still be collidable, leading to it interfering while being invisible
            self.set_position(*deposit.get_position())

            # person should be invisible now that it's successful
            self.deposited=True
            return True, info
        return False, info

    def step(self):
        """ stay coupled to object if any, otherwise nothing """
        if self.grabbed:
            # NOTE: coupled behavior is going to position of one agent arbitrary agent (in this case first)
            #       whether or not the agents are split is up to the controller NOT the Person

            # change position to follow coupled agent
            first_key=list(self.coupled.keys())[0]
            agnt=self.coupled[first_key]
            self.set_position(*agnt.get_position())

        self._clock+=1
        return True


"""
Abstract Agent class

Inventory 2 objects maximum : [slot, slot]
    -picking up a person requires 1 slots

actions:
    PickUpSupply; StoreSupply - only success if agent and reservoir in same position
    UseResource(resource) - if inventory has two types or one (correct type), only use one. otherwise, fail action (w/ feedback, no 'resource' in inventory)

NOTE: think about representation of inventory for the llm
"""

@with_id
@named
@with_position(mutable=True)
@collidable(is_collidable=True)
class AbsAgent:
    """ Abstract agent class (no ambiguity resolution, or other complex behavior), only data & simple interactive functions """
    """ NOTE (important implementation detail) if person in inventory, all other slots are cleared and full """

    INVENTORY_CAPACITY=3
    TYPES=['A', 'B']
    PERSON='PERSON'

    def __init__(self, inventory=None):
        if inventory is None:
            # inventory=defaultdict(int)
            inventory=dict([(k,0) for k in AbsAgent.TYPES])
        else:
            upper_keys(inventory)
            assert set(inventory.keys())==set(AbsAgent.TYPES), f"Wrong keys for inventory. Has: {inventory.keys()}, should have: {AbsAgent.TYPES}"

        # NO PERSON is allowed on inventory @ init
        inventory[AbsAgent.PERSON]=0

        self.inventory=inventory
        assert self.available>=0, f"Too many objects in inventory {self.inventory}"

    @staticmethod
    def set_capacity(capacity):
        AbsAgent.INVENTORY_CAPACITY=capacity

    @property
    def has_person(self):
       return self.inventory[AbsAgent.PERSON]>0

    def used_space(self, tp : str = None):
        if tp is None:
            return sum(self.inventory.values())
        tp=tp.upper()
        assert tp in AbsAgent.TYPES, f"Wrong type for inventory. Tried: {tp}, should have: {AbsAgent.TYPES}"
        return self.inventory[tp]

    @property
    def occupied(self):
        return self.used_space()

    @property
    def available(self):
        return AbsAgent.INVENTORY_CAPACITY-self.occupied

    @property
    def full(self):
        return self.available==0

    def clear_inventory(self, including_person=False):
        cleared=0
        for k,v in self.inventory.items():
            if not including_person and k==AbsAgent.PERSON:
                continue
            cleared+=self.inventory[k]
            self.inventory[k]=0
        return cleared

    def add_inventory(self, tp : str, amt : int = 1):
        """ Failure if not enough space, otherwise add """
        tp=tp.upper()
        # forcing to not add Person since it can only be added through the person instance itself (avoid confusion)
        assert tp in AbsAgent.TYPES, f"Type to add to inventory must be of the following: {AbsAgent.TYPES}"

        if self.available>=amt:
            self.inventory[tp]+=amt
            return True
        return False

    def use_inventory(self, tp : str, amt : int = 1):
        """ Failure if tp is empty, otherwise use"""
        tp=tp.upper()
        # forcing to not use inventory for Person since it can only be managed through the person instance itself (avoid confusion)
        assert tp in AbsAgent.TYPES, f"Type to add to inventory must be of the following: {AbsAgent.TYPES}"

        if self.used_space(tp)>=amt:
            self.inventory[tp]-=amt
            return True
        return False

    def deposit_all_inventory(self, deposit):
        """ Both .use_inventory(.) and deposit.store(.) """
        # NOTE: this function also takes into account interactability
        info={
            'interactable' : None,
            }
        close=deposit.sees(self)
        info['interactable']=close

        if close:
            # copy over dictionary w/o the PERSON key
            inventory_fn=copy.deepcopy(self.inventory)
            inventory_fn[AbsAgent.PERSON]=0

            deposit_success=deposit.store_all(inventory_fn)
            if not deposit_success: # not enough space
                return False,info
            # else clear inventory
            self.clear_inventory(including_person=False)
            return True,info

        return False,info

    """ functions used by Person (when picked up); otherwise they are unaccesible """
    def _add_person(self, person):
        """ NEVER add person manually (only done by Person class) """
        # False if another person is already being carried
        if not self.has_person:
            # adding person drops all other items - not including person
            self.clear_inventory(including_person=False)
            # adding person takes entirety of space
            self.inventory[AbsAgent.PERSON]=AbsAgent.INVENTORY_CAPACITY
            return True
        return False

    def _remove_person(self):
        """ NEVER remove person manually (only done by Person class) """
        # False if not person is being carried
        if self.has_person:
            self.inventory[AbsAgent.PERSON]=0
            return True
        return False


"""
Field class

-reservoirs - list
-deposits - list
-agents - list
-persons - list
-fires - list
-geography - (? some representation)

interface w/ both engine and env (this is the shared representation between them)

visibility to agent vs not (w/ memory, i.e. previously visible -> visible)
"""

class Field:
    """
    Keeps all the POIs and agents in one cohesive class.
    -Tracks visibility - which objects are initially visible + which are currently visible
    -Gives partial observation given visibility
    -Takes .step() actions for all objects

    No actions taken, that is done by Env class (actions are)

    CONDITIONS:
        -Many functions are cached because it is assumed that Field will remain static (no dynamic addition of objects)
        -All objects are assumed to have names
        -An objects name is also not assumed to change
    """

    RECURSIVE_CLASSES=['Fire']
    
    CLASS_TYPES=['Flammable', 'Fire', 'Person', 'Reservoir', 'Deposit', 'AbsAgent']

    # mappers from deposit readable to llm readable - all capital
    READABLE_TYPE_MAPPER_RESOURCE={Deposit.TYPES[0] : "SAND", Deposit.TYPES[1] : "WATER", Deposit.TYPES[2] : "PERSON"}
    UNREADABLE_TYPE_MAPPER_RESOURCE=dict([(v,k) for k,v in READABLE_TYPE_MAPPER_RESOURCE.items()])
    PERSON=Deposit.PERSON

    READABLE_TYPE_MAPPER_FIRE={Flammable.TYPES[0] : "CHEMICAL", Flammable.TYPES[1] : "NON-CHEMICAL"}
    UNREADABLE_TYPE_MAPPER_FIRE=dict([(v,k) for k,v in READABLE_TYPE_MAPPER_FIRE.items()])

    def __init__(self, agents=[], persons=[], reservoirs=[], deposits=[], fires=[]):
        # in order of increasing rendering priority (e.g reservoirs are rendered on top of fires
        self.map={
                'fires' : fires,
                'reservoirs' : reservoirs,
                'deposits' : deposits,
                'agents' : agents,
                'persons' : persons,
                }

        # ----- mapping -----
        # mapping from id to object for easy search later on; static since id doesn't change
        self.id_mapping=[]
        for poi,l in self.map.items():
            for idx,o in enumerate(l):
                # includes Fire
                self.id_mapping.append( ( o.id, o ) )

                # recursive enumeration - make sure that we have mapping for inner objects too!
                if o.class_name() in Field.RECURSIVE_CLASSES:
                    objs=o.all_objects(expand=True, with_memory=False)
                    for io in objs: # io = inner object
                        self.id_mapping.append( ( io.id,  io )  )

        self.id_mapping=dict(self.id_mapping)

        # mapping from name to id for easy search later on; static since name shouldn't doesn't change!
        self.name_mapping=self.all_names(expand=True, dct=True)

        empty_map_d=lambda : dict([(k,[]) for k in self.map.keys()])
        self.visibility = dict([(i,empty_map_d()) for i in range(len(agents))]) # no default dict as we want to throw error
        # --------------------



        # ----- visibility ------
        # visibility dict structure:
        # {
        # agent_id : {'agents' : [1,3], ..., 'persons' : []}
        # }
        # has POI str : list with indexes to visible POIs in the self.map dict
        # fires are initially impotent, make potent here at beginning by making them visible
        #       NOTE: to make them initially not visible and only potent when seen remove these lines
        #           for now we're assuming Persons is the only task requiring search (the other is 'disaster management')
        # also, make all deposits initially visible
        # NOTE: the portion of reservoirs that are initially visible is determined by the procedural generator, default is hidden

        # NOTE: Fires ARE visible initially
        self._initially_visible_pois=['fires', 'deposits', 'reservoirs']
        for poi in self._initially_visible_pois:
            self._make_allof_poi_visible(poi)

        self.update_visibility()

    def _make_allof_poi_visible(self, poi : str):
        for agent_id, visd in self.visibility.items():
            visd[poi]+=[i for i in range(len(self.map[poi]))]

    @functools.cache
    def get(self, poi : str, idx : int = None):
        poi=poi.lower().strip()
        assert poi in self.map.keys(), f"POI: {poi}, must be one of the following: {self.map.keys()}"
        poi_l=self.map[poi]
        if idx is None:
            return poi_l
        return poi_l[idx]

    @functools.cache
    def id_get(self, _id):
        """ Get object from id """
        o=self.id_mapping.get(_id, None)
        return o

    @functools.cache
    def get_id(self, nm):
        """ Get id from name """
        oid=self.name_mapping.get(nm, None)
        return oid

    @functools.cache
    def name_get(self, nm):
        """
        Map name to object, trivial for all other objects except
        Fire types. For a fire, its own mapping method will be called so that
        we can add subtleties about named positions within the fire (such as southof+fire_name),
        the fire methods will return flammable objects.
        """
        oid=self.get_id(nm)
        return self.id_get(oid)

    @functools.cache
    def all_names(self, expand=True, dct=False):
        """ get list of names (can be recursive, e.g. Fire->fire+flammable) """
        # NOTE: All names also includes the agent objects.
        objs=self.all_objects(expand=expand, with_memory=expand) # does include Fire AND Flammable
        objs=list(filter(lambda o : o.get_name() is not None, objs)) # filter for objects w/ no name

        objs_name_dict=dict([(o.get_name(), o.id) for o in objs])
        if dct:
            return objs_name_dict
        return list(objs_name_dict.keys())

    @functools.cache
    def all_objects(self, expand=False, with_memory=False):
        objs=[]
        for poi,l in self.map.items():
            if expand and (len(l)>0) and (l[0].class_name() in Field.RECURSIVE_CLASSES):
                for o in l:
                    ll=o.all_objects(expand=expand, with_memory=with_memory)
                    objs+=ll
                # if we do continue here, we add ONLY Flammables (not Fire) when expanded
                # otherwise (when with_memory is true), we also add the object itself AND its constituents
                if not with_memory: continue
            objs+=l
        return objs 
             

    def update_visibility(self):
        """
        Update visibility for all the agents
        -doesn't include own agent as visible (but includes all others)
        -doesn't include obstacles (from engine)

        Objects remain "visible" for their entire lifetime, EXCEPT if in overwrite_visibility (like Person)
        """

        # object for which to not have previous visibility imply current
        # do for person (note that person has 'spotted' feature, so .sees(.) is different), and agent
        overwrite_visibility=['persons', 'agents']

        for poi, objs in self.map.items():

            # overwrite visibility - if necessary
            if poi in overwrite_visibility:
                for _ in range(len(self.map['agents'])): self.visibility[_][poi]=[]

            for idx,obj in enumerate(objs):
                for i,a in enumerate(self.map['agents']):
                    visd=self.visibility[i]
                    # notice that we add to visible if obj can 'see' us (i.e. POI is within its radius of visibility)

                    # NOTE: This doesn't include itself as visible - due to the trivially false .sees(.)
                    obj_visible=obj.sees(a)

                    if obj_visible and (idx not in visd[poi]):
                        visd[poi].append(idx)

        # make all visible fires potent
        visible_fr_set=[]
        for i,a in enumerate(self.map['agents']):
            visible_fr_set+=self.visibility[i]['fires']
        visible_fr_set=set(visible_fr_set)

        for i,fr in enumerate(self.map['fires']):
            if i in visible_fr_set:
                fr.make_potent()

    def partial_observation(self, agent_idx : int, expand : bool = False):
        """ Get abstract partial visual surroundings 
         Expands makes the observation recursive (adds nameable)

         returns set of visible objects (NOT index to objects in map).
         """

        # update visibility when partial obs is needed (in case outer changes outside of .step() are done)
        self.update_visibility()

        obs=[]
        for poi,idxs in self.visibility[agent_idx].items():
            for idx in idxs:
                o=self.map[poi][idx]
                obs.append(o)

                if expand and o.class_name() in Field.RECURSIVE_CLASSES:
                    # get all objects recursively
                    objs=o.all_objects(expand=expand, with_memory=False)
                    # NOTE: only add expanded/recursive objects if they have a name (i.e. they're in the our names dict)
                    has_name=lambda o : o.name in self.name_mapping.keys()
                    objs=list(filter(has_name, objs))
                    obs+=objs

        return obs

    def local_partial_observation(self, agent_idx : int, diagonal : bool = True):
        """ Get concrete local observation (up,down,left,right,diagonals)
        of which objects are at each position.

        Is recursive (as granular as possible), with_memory=False as we want to avoid abstraction in local obs
        returns dictionary with delta from agent (e.g. (-1,1)) and list of the objects in that position

        NOTE: Only use this if you need textual input for visuals (otherwise, it is redundant)
        """
        agent=self.get('agents', agent_idx)

        dct={}

        dn=GPS.around(central_position=agent.position, layer=1, inclusive=True, diagonal=True)
        # filter out objects not in field AND not current agent AND not of type RECURSIVE_CLASSES (since those are abstractions & are not locally visible)
        for k,l in dn.items():
            # NOTE: we don't include the z axis here (not necessary for partial observation delta)
            deltak=Coordinate.delta(Coordinate(*k),agent.position)[:2]
            dct[deltak]=[]
            for v in l:
                if ((v not in self.id_mapping.keys()) or (v==agent.id)): continue
                o=self.id_mapping[v]
                # not added if in RECURSIVE_CLASSES - fire
                if (o.class_name() in Field.RECURSIVE_CLASSES): continue

                dct[deltak].append(o)

        return dct


    def step(self):
        """
        Does .step() on all the POIs that require it (fire, person, etc)
        .step() doesn't exist & not done for agents (this is to be done by the engine)
        """
        for poi,l in self.map.items():
            if len(l)>0 and hasattr(l[0], 'step') and callable(l[0].step):
                for obj in l: obj.step()

        # DON'T update visibility in .step() - NOTE: AND also do lazily in "partial_observation"
        # update visibility when partial obs is needed (in case outer changes outside of .step() are done)
        # self.visibility should not be used without using "partial_observation" first (which it SHOULDN'T)

        # TODO: what to use for state update w/ engine
        #       add back update visibility here?
        # self.update_visibility()

    @staticmethod
    def procedural_generation(params : dict, seed : int = 42):
        """
        Procedural generation class for this map.


        -grid_size: (width, height, altitude)


        *NOTE: position,name is a parameter given for all
        -reservoirs : list with (tp) (infinite capacity)
        -deposits : list with () (nothing extra required, infinite capacity default)
        -fires: list with (tp,amt_light,enclosing_grid) - params are params needed for proc. gen. for fire (recall position is topleft)
        -persons: list with (extra_load)
        -agents: list with () (nothing extra is required)
        
        NOTE: All reservoirs AND deposits have infinite capacity (see rationale in controller class comments)


        Including generation of "geography"
            -NOTE: this geography can be imported from above (e.g. from engine-like system), but ultimately it's created here
            -This objects must be collidable
        Generate names for the main classes - fire will have the flammable name generation in it's own proc. gen. function
        Calibration for radius size for each POI 
            -not for fire, it has own .procedural_generation() fn, increasing depending on person(s) (group) size?
            -Prevent overlapping radii (to avoid ambiguity in drop/use etc. actions) - but it should be fine because .step(.) assumes you give it id of obj to use/drop it to
            -make the radius of the agents large enough to fit another agent in - if that functionality is desired
                -radius should dictate where should navigateto be successful
        Creation of fire shape & pinch-points
        Make sure that all reservoirs AND deposits have infinite capacity (see rationale in controller class comments)
        Proper scaling to multiple agents (amount of deposits, reservoirs, etc)
        """
        # Flammable, Fire, Person, Reservoir, Deposit, AbsAgent
       
        # NOTE: engine is one of fn arguments
        # - need information about obstacles/geography from engine (can only be done w/ engine)
        # NOTE: need to make fire's .procedural_generation() function

        mapw,maph,mapal=params.pop('grid_size')
        area=mapw*maph
        Coordinate.set_params(width=mapw,height=maph,altitude=mapal)
        # @here
        # TODO: include z axis exclusively when doing initialization (otherwise keep at xy)
        Coordinate.set_axes_mode('xyz')
        
        corners=[(x,y) for x,y in itertools.product([0,mapw-1],[0,maph-1])]

        field_params={}

        for poi,ll in params.items():
            field_params[poi]=[]
            for args in ll:
                if poi=='reservoirs':
                    # args -> tp,position,name
                    # NOTE: give infinite capacity
                    o=Reservoir(args.tp, math.inf, position=args.position)
                    o.set_radius(3*math.sqrt(2)) # 2-layer neighboors including diagonal
                    o.set_name(args.name)
                elif poi=='deposits':
                    # args -> position,name
                    # NOTE: give infinite capacity
                    o=Deposit(capacity=math.inf, position=args.position)
                    o.set_radius(3*math.sqrt(2)) # 2-layer neighboors including diagonal (more than this requires another)
                    o.set_name(args.name)
                elif poi=='fires':
                    # args -> tp,amt_light,enclosing_grid,name,position (topleft)
                    max_w,max_h=args.enclosing_grid
                    o=Fire.procedural_generation(fire_type=args.tp, amt_light=args.amt_light, amt_regions=args.amt_regions, max_w=max_w, max_h=max_h, proportion_filled=.85, top_left=args.position, fire_name=args.name, seed=seed, shape='circle')
                    # *radius already set
                    # *name already set
                elif poi=='persons':
                    # args -> extra_load,position,name
                    o=Person(extra_load=args.extra_load, position=args.position)
                    o.set_name(args.name)

                    # NOTE: takes up ~FIND_PROBABILITY% of map area (if located in the middle)
                    # linear interpolation between max distance from one of corners (as radius)
                    #                              to 0
                    # even though it's not linear
                    if hasattr(args, 'find_probability'):
                        FIND_PROBABILITY=args.find_probability
                        assert 0<FIND_PROBABILITY<=1, f"Find probability for person {o.get_name()} not within (0,1]"
                    else: FIND_PROBABILITY=.20
                    max_corner_distance=max([Coordinate.euclidean(o.position, Coordinate(*corner)) for corner in corners])

                    o.set_radius(FIND_PROBABILITY*max_corner_distance)
                elif poi=='agents':
                    # args -> position,name
                    o=AbsAgent(position=args.position)
                    o.set_radius(3*math.sqrt(2)) # 2-layer neighboors including diagonal
                    o.set_name(args.name)


                field_params[poi].append(o)
        field=Field(**field_params)

        return field


class Controller:

    # DEFINE : actions that can be taken
    MOVEMENT_ACTIONS=['NavigateTo', 'Move']
    CARRY_DROP_ACTIONS=['Carry', 'DropOff']
    SUPPLY_ACTIONS=['StoreSupply', 'UseSupply', 'GetSupply', 'ClearInventory']
    ALL_ACTIONS=MOVEMENT_ACTIONS+CARRY_DROP_ACTIONS+SUPPLY_ACTIONS

    # --- from field ---
    # class types
    CLASS_TYPES=Field.CLASS_TYPES
    # supply types
    SUPPLY_TYPES=list(filter(lambda k : k!=Field.PERSON, Field.UNREADABLE_TYPE_MAPPER_RESOURCE.keys()))
    # ---            ---

    # NOTE: change to True if want to sub-divide the fire into regions
    FIRE_SUBDIVISION=True
    # NOTE: change to True if you want the description of the fire (to the llm) to say the type of fire it is
    TELL_FIRE_TYPE=True
    # NOTE: change to True if you want to filter out the regions that have no objects on fire
    FILTER_REGIONS=True

    # DEFINE: cardinal directions Up,Down,Left,Right and corners
    CARDINAL_VERTICAL={
            'Up' : (0,-1), # flipped
            'Down' : (0,1), # flipped
            }
    CARDINAL_HORIZONTAL={
            'Left' : (-1,0),
            'Right' : (1,0),
            }

    # NOTE: This is used for movement (simplified version, if you want more then change this)
    #           include the center
    MOVABLE_CARDINAL_DIRECTIONS=list(CARDINAL_VERTICAL.keys())+list(CARDINAL_HORIZONTAL.keys())+['Center']

    TO_DELTA_MAP=[(k,v) for k,v in CARDINAL_VERTICAL.items()]+[(k,v) for k,v in CARDINAL_HORIZONTAL.items()]
    for kvert,vvert in CARDINAL_VERTICAL.items():
        for khorz, vhorz in CARDINAL_HORIZONTAL.items():
            TO_DELTA_MAP.append((kvert+khorz, tuple_add(vvert,vhorz)))
    TO_DELTA_MAP.append(('Center', (0,0)))
    TO_DELTA_MAP=dict(TO_DELTA_MAP)

    FROM_DELTA_MAP=dict([(v,k) for k,v in TO_DELTA_MAP.items()])

    # last known initialization parameters,
    # have this as a global parameter so it can be accessed when creating possible actions (using obj names)
    # assumption: if we create multiple controllers, they will have the same env initialization
    #             otherwise this does not work
    LAST_INIT_PARAMS=None

    @staticmethod
    @deprecated() # not supported - create Scenes/scene_{i} file instead
    def pg_params(agent_names=['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Finn'], scene=None):
        num_agents=len(agent_names)

        # 30 x 30, contains one of all w/ n_agents
        # each fire has 2 initially lighted locations
        # names: ReservoirUtah, ReservoirYork, DepositFacility, CaldorFire, GreatFire, LostPersonTimmy, Alice, Bob, Charlie, David, Emma, Finn

        if scene is None: scene=1
        if scene==1:
            # area_900_allones_default initialization
            assert 1<=num_agents<=6, f"For the 'area_900_allones_default' initialization min: 1 and max: 6 are supported not {num_agents}"
            params={
                    'grid_size' : (30,30,1),

                    'reservoirs' : [
                        Arg(tp='a',name='ReservoirUtah',position=(15,5)),
                        Arg(tp='b', name='ReservoirYork',position=(18,8))
                        ],
                    'deposits' : [
                        Arg(name='DepositFacility', position=(23,12)),
                        ],
                    'fires' : [
                        Arg(tp='a', amt_light=2, amt_regions=None, enclosing_grid=(6,6), name='CaldorFire', position=(2,2)),
                        Arg(tp='b', amt_light=2, amt_regions=None, enclosing_grid=(6,6), name='GreatFire', position=(20,16))
                        ],
                    'persons' : [
                        Arg(extra_load=0,name='LostPersonTimmy',position=(5,25)),
                        ],
                    'agents' : [
                        Arg(position=(9,7)),
                        Arg(position=(17,17)),
                        Arg(position=(16,18)),
                        Arg(position=(10,17)),
                        Arg(position=(11,18)),
                        Arg(position=(25,25))
                        ][:num_agents],
                    }
        else:
            raise NotImplementedError

        # name agents
        for i,args in enumerate(params.get('agents',[])):
            args.name=agent_names[i]

        # NOTE (USEFUL): Naming convention. Name must contain class type (except agents)
        for poi,l in params.items():
            if poi in ['grid_size','agents']: continue
            o_type=poi[:-len('s')].capitalize()
            for i,arg in enumerate(l):
                assert o_type in arg.name, f"Object of type {o_type} in initialization does not contain '{o_type}' in name {arg.name}"

        # IMPORTANT - update the last init params global variable - used in feasable action mapping
        Controller.LAST_INIT_PARAMS=params

        return params

    def __init__(self, procedural_generation_parameters, seed=42):
        self.num_agents=len(procedural_generation_parameters['agents'])

        self.field=Field.procedural_generation(params=procedural_generation_parameters, seed=seed)
        assert (len(self.field.get('agents'))==self.num_agents), f"Number of agents provided in parameters {len(self.field.get('agents'))} does not match argument {self.num_agents}"

        # give all the recursive objects w/o fire object (fire is abstraction) so with_memory=False
        self.backend=Backend(objects=self.field.all_objects(expand=True, with_memory=False), engine='grid')
        # otherwise, reset your clock and take a step in the field
        self.clock=dict([(idx, 0) for idx in range(self.num_agents)])
        

    @property
    def all_names(self):
        names=self.field.all_names(expand=True, dct=False)
        # @here
        # since this fn is not cached, we can filter here
        def region_has_fire(nm):
            if '_Region' not in nm:
                return True
            fl=self.name_get(nm)
            fr=self.name_get(fl.parent_name)
            intensity=fr.average_neighboor_intensity(fl)
            return read_enum(intensity).capitalize()!='None'
        names=list(filter(region_has_fire, names))
        return names

    def _monolithic_step(self, agent_idx):
        self.clock[agent_idx]+=1

        # Updates the field environment w/ .step()
        #       this is done after all the agents have finished an action (if not, it will be inert)
        #       NOTE (VERY IMPORTANT): we keep track of the amount of steps by assuming all agents going the a non-zero amt of times is a centralized step
        #           the step action is only for fire (needs to be done after all agents have acted),
        #           and person (will update internally as accordingly, but should move with flock only after all indiv. agents have moved

        # if all the same
        vl=self.clock[0]
        if not (vl>0): return  # inert if 0

        # for agent_idx,clk in self.clock.items():
            # if clk!=vl: return # inert if any diverge (not equal)

        # inert if any of them is 0
        for agent_idx,clk in self.clock.items():
            if clk<1: return
            # if clk!=vl: return # inert if any diverge (not equal)

        # otherwise, reset your clock and take a step in the field
        self.clock=dict([(idx, 0) for idx in range(self.num_agents)])
        self.field.step()

    def step(self, action : str, **action_args):
        """ Wrapper for .raw_step(), allows the step to be inert """
        event=self.raw_step(action, **action_args)

        agent_idx=action_args['agent_idx']
        self.get_observation(agent_idx=agent_idx, dct=event) # updates in place
        # do monolithic step - done after observation
        # reason: if final agent, we don't want its observation to be from the next step.
        # if first agent, we had already updated it after the last of the previous (so it'll have the proper obs already)
        inert_step=action_args.get('inert_step', False)
        if not inert_step: self._monolithic_step(agent_idx)

        return event

    # NOTE: This is .step() w/ no global,local obs & not monolithic_step
    def raw_step(self, action : str, **action_args):
        # NOTE: make sure no valid (even if "wrong") actions give runtime errors
        agent_idx=action_args['agent_idx']
        agent=self.field.get('agents', agent_idx)

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

        Types of error_type(s):
        -restricted_action, not_visible, not_interactable
        """

        event={
            'success' : None,
            'global_obs' : None,
            'local_obs' : None,
            'visual_obs' : None,
            'error_type' : '',
            'info' : '',
            } 

        # LUMP of possible arguments different actions might need
        from_target_id=action_args.get('from_target_id', None)
        to_target_id=action_args.get('to_target_id', None)
        supply_type=action_args.get('supply_type', None)


        # NOTE: Notice order here.
        #       First, we check NoOp, since that always is successful,
        #       Secondly, we check if doing restricted action -- this is because we want to make sure the LLM understands the
        #           underlying, reason for failing: not that object wasn't visible, but that its restricted
        #       THEN, we check if object is within visibility

        # -----------------------------------------------------------------------------
        if action=='NoOp':
            # do nothing - skip to end
            event['success']=True
            return event

        # -----------------------------------------------------------------------------
        # If agent is carrying person, limit actions to movement only (fail all other actions)
        # NOTE: LLM - make this explicit in the rules of the environment
        if agent.has_person and ((action not in Controller.MOVEMENT_ACTIONS) and (action not in Controller.CARRY_DROP_ACTIONS)):
            # skip straight to the end

            event['success']=False
            event['info']='person in inventory; no non-movement actions allowed'
            event['error_type']='restricted_action' # this means they cannot do non-carry-drop/non-movement actions
            return event
        # -----------------------------------------------------------------------------

        # (IMPORTANT) : add difference between not interactable, not visible, or neither in error message (error_type)
        # make sure that navigation does NOT take interactability into account (otherwise we can never interact unless we micro-move there)

        # -----------------------------------------------------------------------------
        # have all objects that are globally visible to agent (so actions can distinguish)
        globally_visible_ids=self.get_globally_visible_ids(agent_idx)
        # both the from and the to have to be visible to the agent!
        # since the id can also be a direction in a special case, then make sure that's handled
        cant_see=lambda _id : (_id is not None) and (_id not in globally_visible_ids) and (_id not in Controller.TO_DELTA_MAP.keys())
        if cant_see(from_target_id) or cant_see(to_target_id):
            # skip straight to end
            # no need to try to perform any actions
            event['success']=False
            event['error_type']='not_visible'
            return event

        # -----------------------------------------------------------------------------

        # movement primitives
        # NavigateTo(obj_id)
        # Move(dx,dy)
        # NOTE: think about movement primitive constraints - moving back after ahead if there is momentum

        movement_actions=Controller.MOVEMENT_ACTIONS
        # NOTE: cannot navigate anywhere if "carrying" person UNLESS there are enough people to carry
        #       -implementation: if person in inventory, you can't do anything

        # -----------------------------------------------------------------------------
        if action in movement_actions:
            if action=='NavigateTo': # AGENT
                # NOTE: You can navigate to another agent if the radius is large enough (since it's not able to be on top of it)
                target_obj=self.id_get(to_target_id)
                eps_radius=target_obj.get_radius()

                success,info=self.backend.navigate(agent, target_obj.position, eps=eps_radius)
                event['success']=success
                # can navigate regardless of radius (only matters if you see the obj & backend factors), so interactability is not important

            elif action=='Move':
                direction=to_target_id
                assert direction in Controller.MOVABLE_CARDINAL_DIRECTIONS, f"Move action called direction {direction} is not valid, must be in {Controller.MOVABLE_CARDINAL_DIRECTIONS}"

                success=self.backend.move(agent, direction)
                event['success']=success
                # interactability not applicable

        # -----------------------------------------------------------------------------
        carry_drop_actions=Controller.CARRY_DROP_ACTIONS
        if action in carry_drop_actions: # PERSON
            person=self.id_get(from_target_id)

            if action=='Carry':
                # use .carry(agent) method
                # Do person.carry(agent) - all else is handled by person
                # *Carry(person)

                if person.class_name()!='Person':
                    event['success']=False
                    event['info']='cannot carry non-person'
                else:
                    success,info=person.pick(agent)
                    event['success']=success
                    # since visibility radius NOT equivalent to how close it is
                    # (for agents other than the first to find it, due to the 'spotted' factor)
                    # then we have to add feedback on whether is it interactable
                    if not info['visible']:
                        event['error_type']='not_interactable' # from .pick(.) method

            elif action=='DropOff':
                # use .drop(agent) method
                # how to give success about DropOff action in an online way without knowing if they have all dropped it off?
                #       -edit it in the class above this one? (take note of this somewhere)
                # *DropOff(deposit_id, person_id)

                # NOTE: deposits should probably have infinite capacity for people (and overall)
                #       they're already bottlenecked due to the get_all rule for getting resources
                #       it's unlikely they should fill it anyway, it's just a buffer zone for resources collected
                deposit=self.id_get(to_target_id)

                if person is None:
                    event['success']=False
                    # failure here should be understood by LLM (since it doesn't have person in inventory)
                elif person.class_name()!='Person':
                    event['success']=False
                    event['info']='cannot drop-off non-person object'
                elif deposit.class_name()!='Deposit':
                    event['success']=False
                    event['info']='cannot drop-off person into non-deposit object'
                else:
                    success,info=person.drop(agent, deposit)
                    event['success']=success
                    if not info['interactable']: # deposit isn't close enough to even be interactable
                        event['error_type']='not_interactable' # from .drop(.) method

    
        # -----------------------------------------------------------------------------
        supply_actions=Controller.SUPPLY_ACTIONS
        if action in supply_actions: # RESERVOIR/DEPOSIT, PERSON
            # if exists supply type, then make it readable to the agent & fire
            if supply_type is not None:
                supply_type=Field.UNREADABLE_TYPE_MAPPER_RESOURCE[supply_type.upper()]

            if action=='StoreSupply':
                # StorySupply - removes from agent, adds to deposit
                # *StoreSupply(resource, deposit_id)
                # Stores ALL of your inventory (not including person)
                deposit=self.id_get(to_target_id)

                if deposit.class_name()!='Deposit':
                    event['success']=False
                    event['info']='cannot drop-off supplies to non-deposit'
                else:
                    success,info=agent.deposit_all_inventory(deposit)
                    # this behavior below is already covered by the restricted_action
                    # no_person=not agent.has_person # can't deposit w/ person
                    if not info['interactable']: # deposit isn't close enough to even be interactable
                        event['error_type']='not_interactable'
                    event['success']=success

            elif action=='UseSupply':
                # *UseSupply(resource_type) - use ALL supply from the inventory of that type
                #       -use supply for ameliorating fire
                fire=self.id_get(to_target_id)

                if fire.class_name()=='Flammable':
                    # get parent fire if we're currently a flammable object
                    fire=self.name_get(fire.parent_name)

                if fire.class_name()!='Fire':
                    event['success']=False
                    event['info']='cannot use supplies for a non-fire'
                else:
                    # When using supplies on fire, use all of it or one unit?
                    # We decided upon only 1 unit since it gives a time-buffer for the resource collector to collect multiple
                    # Useful in the encouragement of multi-agent collaboration
                    # Furthermore, it gives the agent more fine control (since fire might have varying intensities, so it'd be a waste to use all)

                    interactable=fire.sees(agent)

                    if interactable:
                        use_success=agent.use_inventory(tp=supply_type, amt=1)
                        if use_success:
                            lessen_success=fire.lessen(loc=agent.position, extinguisher_type=supply_type, diagonal=True)

                        # NOTE: Success is only measured here by whether it was dropped (i.e. supply exists) & it lessened at least 1 flammable,
                        #       NOT whether the fire is still raging on
                        event['success']=use_success and lessen_success
                    else:
                        event['error_type']='not_interactable'
                        event['success']=False

            elif action=='GetSupply':
                # NOTE: differential rate limiting - artificial rate limit of collection : 1-unit for reservoir, any for deposit
                #       This rate limiting is the reason why deposits are useful
                # *GetSupply(resource_type, deposit_id/reservoir_id) - if resource_type is None (or non-existant index), then must be reservoir
                dropoff=self.id_get(from_target_id)
                interactable=dropoff.sees(agent)

                if dropoff.class_name()=='Deposit':
                    agent_space=agent.available # depends on amt of materials & size of inventory (default is 2)

                    # if within radius
                    if interactable:
                        resource_extracted=dropoff.use(supply_type, agent_space)
                        add_success=agent.add_inventory(tp=supply_type, amt=resource_extracted)
                        use_success=(resource_extracted>0)
                        event['success']=use_success and add_success
                    else:
                        event['success']=False
                        event['info']='Not within radius of deposit'
                        event['error_type']='not_interactable'

                elif dropoff.class_name()=='Reservoir':
                    # if within radius
                    if dropoff.sees(agent):
                        # Uses exactly 1 from reservoir (should have infinite capacity)
                        supply_type=dropoff.type
                        used=dropoff.use(1)
                        get_success=(used>0)
                        assert get_success, 'Reservoir not of infinite capacity, update this fn to work with finite capacity ones.'
                        if get_success:
                            add_success=agent.add_inventory(tp=supply_type, amt=1)
                        event['success']=get_success and add_success
                    else:
                        event['success']=False
                        event['info']='Not within radius of reservoir'
                        event['error_type']='not_interactable'

                else:
                    event['success']=False
                    event['info']='cannot get supplies from non-deposit/non-reservoir source'

            elif action=='ClearInventory':
                # Clear the agent's inventory absolutely - unless there is a person
                # Do this action for clearing to pick up resources
                clear_amt=agent.clear_inventory(including_person=False)
                person_success=(not agent.has_person)
                # Fail if there is a person in inventory
                event['success']=person_success

        # -----------------------------------------------------------------------------

        # returns - event (dict, w/ info on error + success)
        return event

    def id_get(self, _id):
        return self.field.id_get(_id)

    def name_get(self, nm):
        return self.field.name_get(nm)

    def get_observation(self, agent_idx, dct=None):
        if dct is None: dct={}
        dct['global_obs']=self.partial_observation(agent_idx)
        dct['local_obs']=self.local_partial_observation(agent_idx)
        return dct

    def get_globally_visible_ids(self, agent_idx):
        dct=self.get_observation(agent_idx)
        gobs=dct['global_obs']
        gobs_ids=list(map(lambda d : d['id'], gobs))
        return gobs_ids

    def _wrap_object_readable(self, obj):
        """
        Wraps object into readable format (dict) w/ relevant information filtered out,
        We do this in order to avoid the top-level system interacting w/ the POI/agent classes directly (abstraction)

        Object types:
        -Flammable, Fire, Person, Reservoir, Deposit, AbsAgent

        Given information:
        {
        # FOR ALL
        'position' : x,y position of the object,
        'name' : given name for object,
        'id' : id of the object,
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

        # FOR ALL
        'string_description' : string readable description of all this information above
        """
        dct={}

        # independent characteristics
        ptpl=obj.get_position()
        dct['position']={axn : axv for axn,axv in zip(['x','y','z'][:len(ptpl)], ptpl)}
        dct['name']=obj.get_name()
        dct['id']=obj.id
        dct['type']=obj.class_name()
        dct['collidable']=obj.collidable
        # dct['string_description']=f"{dct['type']} named {dct['name']}"
        dct['string_description']=f"{dct['name']}"

        tp=obj.class_name()
        fire_type_desc=lambda tp : f" of {tp} type" if Controller.TELL_FIRE_TYPE else ""

        if tp == 'Flammable':
            # NOTE: if flammable has reached this fn, it must have name
            assert hasattr(obj, 'name'), f"Flammable object in observation yet has no name"

            dct['parent_fire']=obj.parent_name
            dct['intensity']=read_enum(obj.intensity).capitalize() # 'none', 'low', 'medium', 'high'
            dct['fire_type']=Field.READABLE_TYPE_MAPPER_FIRE[obj.fire_type].capitalize() # 'chemical' or 'non-chemical'
            fire=self.name_get(dct['parent_fire'])
            average_neighboor_intensity=fire.average_neighboor_intensity(obj)
            dct['average_neighboor_intensity']=read_enum(average_neighboor_intensity).capitalize() # 'none', 'low', 'medium', 'high'

            # get average intensity of neighboor, instead of specific point, to give more representative information about 'region'
            dct['string_description']+=f" with an intensity of {dct['average_neighboor_intensity']}"
            dct['string_description']+=fire_type_desc(dct['fire_type'])

        elif tp == 'Fire':
            dct['average_intensity']=read_enum(obj.average_intensity).capitalize() # 'none', 'low', 'medium', 'high'
            dct['fire_type']=Field.READABLE_TYPE_MAPPER_FIRE[obj.fire_type].capitalize() # 'chemical' or 'non-chemical'

            dct['string_description']+=f" with average intensity of {dct['average_intensity']}"
            dct['string_description']+=fire_type_desc(dct['fire_type'])

        elif tp == 'Person':
            dct['load']=obj.load # number corresponding to amt of agents needed
            dct['status']=read_enum(obj.status).capitalize() # 'grabbed' or 'grounded'
            dct['spotted']=obj.spotted # False or True depending if it has been spotted
            dct['deposited']=obj.deposited # False or True depending if it has been deposited

            if dct['deposited']:
                dct['string_description']+=f" that has been safely deposited, congrats!"
            else:
                dct['string_description']+=f" requiring {dct['load']} agents to carry"

        elif tp == 'Reservoir':
            dct['resource_type']=Field.READABLE_TYPE_MAPPER_RESOURCE[obj.type].capitalize() # 'sand' or 'water'
            dct['inventory']=obj.available # dict w/ {resource_type : amt} (likely math.inf)

            dct['string_description']+=f" containing {dct['resource_type']}"

        elif tp == 'Deposit':
            inventory=dict([  (Field.READABLE_TYPE_MAPPER_RESOURCE[k].capitalize(),v) for k,v in obj.storage.items()  ])
            dct['inventory']=inventory # dict w/ {resource_type : amt} for all resource types (including person)
            
            dct['string_description']+=f" containing {dct['inventory']}"

        elif tp == 'AbsAgent':
            inventory=dict([  (Field.READABLE_TYPE_MAPPER_RESOURCE[k].capitalize(),v) for k,v in obj.inventory.items()  ])
            dct['inventory']=inventory # dict w/ {resource_type : amt} for all resource types (including person)

            dct['string_description']+=f" containing {dct['inventory']}"

        else:
            dct['string_description']=f""
            # fire regions are flammable object, handled there

        return dct

    def get_id(self, name):
        return self.field.get_id(name)

    def get(self, poi : str, idx : int = None):
        return self.field.get(poi=poi, idx=idx)

    def get_inventory(self, agent_idx : int, tp : str = None):
        """ Returns agent inventory as dict (readable format) """
        _inventory=self.get('agents', agent_idx).inventory
        inventory=dict([  (Field.READABLE_TYPE_MAPPER_RESOURCE[k].capitalize(),v) for k,v in _inventory.items()  ])
        if tp is None:
            return inventory
        return inventory.get(tp.capitalize(), None)

    def partial_observation(self, agent_idx : int):
        """
        wraps fn from field
        making it so wraps the raw objects into dictionary formats (see _wrap_object_readable)

        No expansion since we want the information here to be global (we're not doing sub-division of fire into regions)
            -if that behavior is desired, change FIRE_SUBDIVISION
        List, no order.
        """
        pobs=self.field.partial_observation(agent_idx=agent_idx, expand=Controller.FIRE_SUBDIVISION)
        dcts=[self._wrap_object_readable(o) for o in pobs]

        # NOTE: Filter out any regions that don't have any active fires on them (to avoid too many)
        if Controller.FILTER_REGIONS:
            dcts=list(filter(lambda d : d.get('average_neighboor_intensity', '')!='None', dcts))

        return dcts
    
    def local_partial_observation(self, agent_idx : int):
        """
        wraps fn from field
        making it so wraps the raw objects into dictionary formats (see _wrap_object_readable)
        Keys of dictionary are directions (left,up,down,right,etc...)
        """
        l_pobs=self.field.local_partial_observation(agent_idx=agent_idx, diagonal=True)
        # NOTE: No filtering of collidable since we want to include flammables
        d={}

        # l_pobs is dict of delta : list
        for k,v in l_pobs.items():
            direction=Controller.FROM_DELTA_MAP[k]
            if direction not in Controller.MOVABLE_CARDINAL_DIRECTIONS: continue # skip if not movable direction

            vv=list(map(self._wrap_object_readable, v))

            d[direction]=vv
        return d



""" Backend! Handles all the interactions w/ the raw engine for the controller to use """

class Backend:
    """
    Provides interface between the engine (e.g. Nvidia ROS, gridworld, etc) and Controller
    """

    def __init__(self, objects, engine='grid'):
        # all objects given should have position, otherwise this breaks
        assert all([hasattr(o, 'position') for o in objects]), f"At least one object given to backend does not have .position, all objects must have"
        # all objects should have collidability, otherwise navigation breaks
        assert all([hasattr(o, 'collidable') for o in objects]), f"At least one object given to backend does not have .collidable, all objects must have"

        self.objects=objects
        if engine=='grid':
            self.engine=GridEngine()
        else:
            raise NotImplementedError

        self.initialize(self.objects)

    def navigate(self, obj, target_position : Coordinate, eps=None):
        """
        Navigate to the target_position with distance <eps of target (and remain within epsilon, no momentum away from radius vector).
            -Default eps=0 for discrete, 1e-6 * max(width, height) for continuous
        For immutable objects (e.g. Fire, Deposit, Reservoir, etc) don't allow position changes.
        For mutable objects, allow them.
        """ 
        if eps is None:
            w,h=Coordinate.get_params()
            eps=0

        # add context
        info={
                'is_mutable' : None,
            }

        # (engine): engine must have path navigation functionality (from position a to b)
        #               (IMPORTANT) in NavigateTo function, make sure that agent ends up within radius of target POI (success/termination condition)
        #                           Otherwise, all the functionality of interaction with the POI breaks!
        #                (for gridworld specifically) avoid collisions between non-collidable objects
        #                   -: add collidable wrapper
        # (engine): engine must have primitive movement functionality for the agent
        #       -add dry_run functionality (?) (for obstacles)

        # if objects position shouldn't be changed
        info['is_mutable']=obj.mutable_position
        if not obj.mutable_position:
            return False, info

        # if failure then end_position is None (handled by object mover)
        end_position=self.engine.navigation(from_position=obj.position, target_position=target_position, eps=eps)
        success=self.engine.move_object(obj, end_position)

        return success,info

    def move(self, obj, direction : str):
        """
        Moves object in the grid in the following directions
        Directions -> up, down, left, right
        Cannot go to objects w/ collision or out of bounds

        For immutable objects (e.g. Fire, Deposit, Reservoir, etc) don't allow position changes.
        For mutable objects, allow them.
        """

        # if objects position shouldn't be changed
        if not obj.mutable_position:
            return False

        delta=Controller.TO_DELTA_MAP[direction.capitalize()]
        target_position=copy.deepcopy(obj.position)
        target_position.change_position(*delta)

        end_position=self.engine.navigation(from_position=obj.position, target_position=target_position, eps=0)
        success=self.engine.move_object(obj, end_position)
        return success

    def update(self):
        raise NotImplementedError
        
    def initialize(self, objects):
        """ Top-down initialization of engine w/ the objects in the environment """
        # engine initialization
        #       -engine keep info to handle collisions & movement
        #       -cache anything necessary
        self.engine.initialize(objects=objects)
    

class GridEngine:
    def __init__(self):
        pass

    def initialize(self, objects):
        # all objects given should have position, otherwise this breaks
        assert all([hasattr(o, 'position') for o in objects]), f"At least one object given to backend does not have .position, all objects must have"
        # all objects should have collidability, otherwise navigation breaks
        assert all([hasattr(o, 'collidable') for o in objects]), f"At least one object given to backend does not have .collidable, all objects must have"

        self.objects=objects
        self.id_mapping=dict([(o.id,o) for o in self.objects])

    @functools.cache
    def id_get(self, _id):
        o=self.id_mapping.get(_id, None)
        return o

    def navigation(self, from_position, target_position, eps):
        """ 
        Raw navigation from position to another, within <=eps distance of target_position.
        Objects are assumed to have a .collidable attribute

        Usually there'd be search (A*), yet this functions just handles the feasability and
        outputs a feasable position.
        If no feasable position is found, then None is outputted. Otherwise, returns tuple of feasable position

        For grid-world, we can trivially move object positions (see function below).
        We're assuming that even if blocked by collidable objects locally, we can jump over.
        """

        # if target position OR from position is out-of-bounds, then we can't move
        if  ((not Coordinate.within_bounds(*from_position.get()))
                or (not Coordinate.within_bounds(*target_position.get())) ):
            return None

        # if they're equal, then trivially yes
        if from_position.get() == target_position.get():
            return target_position.get()

        # get dct w/ position keys and id list values
        dct=GPS.near(central_position=target_position, radius=eps)
        for pos_xy, idl in dct.items():
            # filter by objects that exist for the engine
            pos_oids=list(filter(lambda oid : oid in self.id_mapping.keys(), idl))
            # get all collidable objects
            collidable_objs=[]
            for oid in pos_oids:
                o=self.id_get(oid)
                assert o is not None, f"Filtered pos_oids by objs in GridEngine yet get None when using dict"
                if o.collidable: collidable_objs.append(o)

            # no objects to collide with then give that location
            if len(collidable_objs)==0:
                # RETURNS TUPLE
                return pos_xy

        # if none were empty within radius, fail
        return None

    def move_object(self, obj, end_position):
        """ For grid-world we trivially tele-transport positions """
        # NOTE: end_position is NOT a Coordinate object
        if end_position is not None:
            obj.set_position(*end_position)
            return True
        return False



# ------------
"""

# ENGINE
# engine must have agent_id to agent mapping (so that it can update the action of the agent w/ agent_id)
# this will be done in the llm-facing env class (movement actions)
# must give (non-textual, non-image) information about the partial observations of the agent (this is later appended w/ map information)
# have engine.step(actions) to update internal physics given low-level actions

Engine class
Similar to physics, collision, geography engine

have grid-world? I only need to handle wall & obstacle collisions (which are equivalent)
    overhead might not be necessary, but neat rendering functions
        -maybe use some of the functionality (core/grid.py)
    why is multigrid faster than grid-world (implement the necessary optimizations)

engine (queries):
    return value

    geography:
        add geography w/ obstacles

        -existsPath(a,b) (navigation)
        -localObs(a, r) - gives partial observation around point a (list of obs/none @ coords), radius r
        -existsObstacle(a) - exists obstacle at position a

engine (actions):
    return success
    updates interal repr

    objects:
        handle collisions

        -moveObject(o, p) - move object o to position (tele-transportation)
            -useful for when getting local observations of agents
            -note two objects can occupy the same position
            -wrapper along with getPosition(o) to moveAhead,back,left,right etc...

"""
