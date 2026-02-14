import unittest
from core import *
from utils import *
from pprint import pprint
import time

class ArgumentTest(unittest.TestCase):
    def test_init(self):
        a=Arg(name='a', position=(0,0), tv=1)
        self.assertEqual(a.name, 'a')
        self.assertEqual(a.position, (0,0))
        self.assertEqual(a.tv, 1)

class CoordinateTest(unittest.TestCase):
    def setUp(self):
        Coordinate.set_axes_mode('xy')
        height, width=10,10
        Coordinate.set_params(width=width, height=height)
        self.eps=1e-6

    # TODO: add sensible non-mode-breaking test for z axis (informally tested w/ env_unittest)

    def test_set_get(self):
        p=Coordinate(1,1)
        q=Coordinate(0,0)
        q.set(*p.get())
        self.assertEqual(p.get(), q.get())

        q.direct_set(*p.direct_get())
        self.assertEqual(p.get(), q.get())

    def test_within_bounds_edges(self):
        # testing the 0-indexed way
        self.assertFalse(Coordinate.within_bounds(10,10))
        self.assertFalse(Coordinate.within_bounds(10,9))
        self.assertFalse(Coordinate.within_bounds(9,10))

        self.assertTrue(Coordinate.within_bounds(9,9))

        # shouldn't give error
        Coordinate(9,9)
        Coordinate(0,0)
        Coordinate(9,0)
        Coordinate(0,9)

    def test_euclidean_distance(self):
        c=Coordinate(0,0)
        d=Coordinate(1,1)
        self.assertTrue(abs(Coordinate.euclidean(c,d) - math.sqrt(2)) < self.eps)

    def test_manhattan_distance(self):
        c=Coordinate(0,0)
        d=Coordinate(2,2)
        self.assertEqual(Coordinate.manhattan(c,d), 4)

    def test_discrete_approximation(self):
        c=Coordinate(2,2)
        self.assertEqual(c.get(), (2,2))

        c=Coordinate(4,6)
        self.assertEqual(c.get(), (4,6))

    def test_within_radius(self):
        r=5 # radius corresponds to 10
        c=Coordinate(0,0,radius=r)

        d=Coordinate(2,4)
        e=Coordinate(2,2)

        # functionality - false (because we want to exclude self-referential behavior)
        self.assertFalse(c.within_radius(c))
        # functionality - trivially true (not same object, so doesn't matter if exact same location)
        import copy
        self.assertTrue(c.within_radius(copy.deepcopy(c)))

        # functionality
        self.assertTrue(c.within_radius(d))
        self.assertTrue(c.within_radius(e))

        # anti-symmetry (since radius of d is 0, so NOT within)
        self.assertFalse(d.within_radius(c))
        self.assertFalse(e.within_radius(c))

    def test_neighboors(self):
        c=Coordinate(1,1)

        dx=1

        d=Coordinate(1+dx,1)
        e=Coordinate(1,1+dx)
        f=Coordinate(1-dx,1)
        g=Coordinate(1,1-dx)

        # not neighboors (as they're diagonal) - diagonal is False by default
        nd=Coordinate(1+dx,1+dx)
        ne=Coordinate(1-dx,1-dx)
        nf=Coordinate(1-dx,1+dx)
        ng=Coordinate(1+dx,1-dx)
        
        self.assertFalse(Coordinate.neighboors(c,nd))
        self.assertFalse(Coordinate.neighboors(c,ne))
        self.assertFalse(Coordinate.neighboors(c,nf))
        self.assertFalse(Coordinate.neighboors(c,ng))

        # change mode to have diagonal
        self.assertTrue(Coordinate.neighboors(c,nd,diagonal=True))
        self.assertTrue(Coordinate.neighboors(c,ne,diagonal=True))
        self.assertTrue(Coordinate.neighboors(c,nf,diagonal=True))
        self.assertTrue(Coordinate.neighboors(c,ng,diagonal=True))


class FlammableTest(unittest.TestCase):
    def test_initialization(self):
        fl = Flammable('a')
        # shouldn't be on fire or spreadable at initialization
        self.assertFalse(fl.on_fire())
        self.assertFalse(fl.spreadable())
        # should be able to light at init
        self.assertTrue(fl.light())

    def test_progression(self):
        fl = Flammable('a')
        fl.light()

        progression=[Intensity.LOW]*Flammable.L_TO_M + [Intensity.MEDIUM]*Flammable.M_TO_H + [Intensity.HIGH]
        for clk in range(len(progression)):
            fl.step()
            self.assertTrue(fl.intensity==progression[clk])

    def test_has_position(self):
        fl = Flammable('a')
        self.assertEqual(fl.get_position(), (0,0))
        fl.set_position(1,1)
        self.assertEqual(fl.get_position(), (1,1))

    def test_lessen_fire(self):
        # w/ type a
        assert len(Flammable.TYPES)>1, 'must have more than one flammable extinguish type for test to work'
        for i,tp in enumerate(Flammable.TYPES):
            fl = Flammable(tp, Intensity.HIGH)
            # use proper type - success, check downgrade in intensity
            self.assertTrue(fl.lessen(tp))
            self.assertEqual(fl.intensity, Intensity.MEDIUM)

            self.assertTrue(fl.lessen(tp))
            self.assertEqual(fl.intensity, Intensity.LOW)

            self.assertTrue(fl.lessen(tp))
            self.assertEqual(fl.intensity, Intensity.NONE)

            self.assertTrue(fl.lessen(tp)) # true even if intensity doesn't lower
            self.assertEqual(fl.intensity, Intensity.NONE)

            # re-ignite to max fire 
            self.assertTrue(fl.light(Intensity.HIGH))

            # use wrong type - failure, check downgrade in intensity
            wtp = Flammable.TYPES[(i+1)%len(Flammable.TYPES)]
            self.assertFalse(fl.lessen(wtp))
            self.assertEqual(fl.intensity, Intensity.HIGH)


class FireTest(unittest.TestCase):

    def setUp(self):
        self.params=Coordinate.get_params()
        Coordinate.set_params(width=3,height=3)

    def tearDown(self):
        Coordinate.set_params(**self.params)

    def initfire(self, light_center=False):
        flammables=[]
        idx=0
        for x in [0,1,2]:
            for y in [0,1,2]:
                fl=Flammable('a', position=(x,y))
                flammables.append(fl)

                if x==1 and y==1:
                    idx=x*3+y

        ctr=flammables.pop(idx)
        flammables.append(ctr) # add fire center at the end
        if light_center:
            ctr.light()
        fire=Fire(flammables, name="jamison fire")
        fire.make_potent()
        return fire

    def test_naming_and_id(self):
        fire=self.initfire()
        fire_2=self.initfire()

        # test naming
        self.assertEqual(fire.get_name(), "jamison fire")

        # test id not equal
        self.assertNotEqual(fire.id, fire_2.id)

    def impotent(self):
        fire=self.initfire(light_center=True, impotent=True)

        # should be potent - due to our init
        self.assertTrue(fire.impotent)
        # success as now is potent
        self.assertTrue(fire.step())


        fire.make_impotent()
        # impotent
        self.assertTrue(fire.impotent)
        # impotent so step() fails
        self.assertFalse(fire.step())

    def test_spread(self):

        fire=self.initfire(light_center=True)
        # render(fire.flammables)
        # to fully spread fire in 3 by 3 starting from center - N
        # 2 initial layers (hence *2), plus final (wait until reach high)
        # +1 because ?
        N=((Flammable.L_TO_M+1)*2+(Flammable.L_TO_M+Flammable.M_TO_H)+1)*Fire.STEP_EVERY
        for clk in range(N):
            fire.step()
            # render(fire.flammables, title=clk)

        # fire should be spread everywhere at max intensity
        for fl in fire.flammables:
            self.assertTrue(fl.on_fire())
            self.assertEqual(fl.intensity,Intensity.HIGH)
            self.assertTrue(fl._clock>Flammable.L_TO_M+Flammable.M_TO_H)

        # ---- test adding ----
        params=Coordinate.get_params()
        Coordinate.set_params(height=4,width=4)

        fire=self.initfire(light_center=True)
        # this flammable should have neighboors
        self.assertTrue(fire.add_flammable(Flammable('a', position=(2,3))))
        # print(dict([(fire.id_get(k).get_position(),list(map(lambda x : x.get_position(),v))) for k,v in fire.dneighboor.items()]))

        # run again - TODO: update the time here
        N=((Flammable.L_TO_M+1)*(2+1)+(Flammable.L_TO_M+Flammable.M_TO_H)+(1+1))*Fire.STEP_EVERY
        for clk in range(N):
            # render(fire.flammables)
            fire.step()

        # fire should be spread everywhere at max intensity
        for fl in fire.flammables:
            self.assertTrue(fl.on_fire())
            self.assertEqual(fl.intensity,Intensity.HIGH)
            self.assertTrue(fl._clock>Flammable.L_TO_M+Flammable.M_TO_H)

        Coordinate.set_params(**params)

    def test_lessen(self):
        fire=self.initfire(light_center=True)
        
        # spread fire everywhere
        N=((Flammable.L_TO_M+1)*2+(Flammable.L_TO_M+Flammable.M_TO_H)+1)*Fire.STEP_EVERY
        for clk in range(N):
            fire.step()

        # fire should be spread everywhere at max intensity
        for fl in fire.flammables:
            self.assertTrue(fl.on_fire())
            self.assertEqual(fl.intensity,Intensity.HIGH)

        ctr=fire.flammables[-1]
        pos=ctr.position

        # lessen fire at center - success
        self.assertTrue(fire.lessen(pos, 'a'))
        # check lessen lowered it by a tick - to medium
        for fl in fire.flammables:
            self.assertEqual(fl.intensity, Intensity.MEDIUM)

        # lessen fire at center - failure
        self.assertFalse(fire.lessen(pos, 'b'))
        # check lessen remained in same position - at medium
        for fl in fire.flammables:
            self.assertEqual(fl.intensity, Intensity.MEDIUM)

        # lessen fire at center - success
        self.assertTrue(fire.lessen(pos, 'a'))
        # check lessen lowered it by a tick - to LOW 
        for fl in fire.flammables:
            self.assertEqual(fl.intensity, Intensity.LOW)

        # lessen fire at center - success
        self.assertTrue(fire.lessen(pos, 'a'))
        # check lessen lowered it by a tick - to NONE!
        for fl in fire.flammables:
            self.assertEqual(fl.intensity, Intensity.NONE)

    def test_procedural_generation(self):
        # This test is mostly visual confirmation
        params=Coordinate.get_params()
        Coordinate.set_params(height=30,width=30)
        Controller.FIRE_SUBDIVISION=True

        top_left=(5,5)
        max_w,max_h=20,20
        amt_light=10
        fire_name="big_fire"

        a=time.time()
        fire=Fire.procedural_generation(fire_type='a', amt_light=amt_light, amt_regions=None, max_w=max_w, max_h=max_h, proportion_filled=.9, top_left=top_left, fire_name=fire_name, seed=42, shape='circle')
        # print('time for proc gen fire', time.time()-a)
        self.assertEqual(sum([f.on_fire() for f in fire.flammables]), amt_light)

        # naming of sub-fires
        cnt=0
        for f in fire.flammables:
            if f.on_fire():
                self.assertEqual(f.get_name(), f"{fire_name}_Region_{cnt+1}")
                cnt+=1

        fire.make_potent()
        # render(fire.flammables)
        for _ in range(20):
            fire.step()
            # render(fire.flammables)

        Coordinate.set_params(**params)
        Controller.FIRE_SUBDIVISION=False

class ReservoirTest(unittest.TestCase):
    def test_use(self):
        tot=10
        r=Reservoir('a', tot)
        self.assertEqual(r.type, 'A')

        self.assertTrue(r.use(5))
        self.assertEqual(r.available, 5)
            
        self.assertTrue(r.use(5))
        self.assertEqual(r.available, 0)

        self.assertFalse(r.use(5))
        self.assertEqual(r.available, 0)

        self.assertTrue(r.empty)

class DepositTest(unittest.TestCase):
    def init(self):
        d={'a' : 6, 'b' : 4, 'person' : 1}
        return Deposit(capacity=11, storage=d)

    def initinf(self):
        d={'a' : 0, 'b' : 0, 'person' : 2}
        return Deposit(storage=d)

    def test_inf(self):
        d=self.initinf()

        self.assertEqual(d.space, math.inf)
        self.assertEqual(d.available('person'), 2)

        # after storing, you should have the same amount
        d.store('person', 2)
        self.assertEqual(d.space, math.inf)
        self.assertEqual(d.available('person'), 4)

    def test_init(self):
        d=self.init()
        self.assertEqual(d.space, 0)
        self.assertEqual(d.available('a'), 6)
        self.assertEqual(d.available('b'), 4)
        self.assertEqual(d.available('person'), 1)

        d=Deposit(capacity=9)
        self.assertEqual(d.space, 9)
        self.assertEqual(d.available('a'), 0)
        self.assertEqual(d.available('b'), 0)

    def test_use(self):
        d=self.init()

        self.assertEqual(d.use('a', 5), 5)
        self.assertEqual(d.available('a'), 1)

        self.assertEqual(d.use('a', 5), 1)
        self.assertEqual(d.available('a'), 0)

        self.assertEqual(d.use('b', 5), 4)
        self.assertEqual(d.available('b'), 0)

    def test_store(self):
        d=self.init()
        d.capacity=21

        self.assertEqual(d.store('a', 5), 5)
        self.assertEqual(d.available('a'), 11)

        self.assertEqual(d.store('b', 5), 5)
        self.assertEqual(d.available('b'), 9)

        self.assertEqual(d.store('b', 5), 0)
        self.assertEqual(d.available('b'), 9)

        # test storing all
        d=self.init()
        d.capacity=11+10
        self.assertTrue(d.store_all({'a' : 4, 'b' : 4, 'person' : 0}))
        self.assertEqual(d.available('a'), 10)
        self.assertEqual(d.available('b'), 8)
        self.assertEqual(d.space, 2) # 10-8=2 spaces left

        self.assertFalse(d.store_all({'a' : 4, 'b' : 4, 'person' : 0}))
        self.assertEqual(d.available('a'), 10)
        self.assertEqual(d.available('b'), 8)
        self.assertEqual(d.space, 2) # 10-8=2 spaces left


class PersonTest(unittest.TestCase):
    def init(self):
        return Person(extra_load=1, radius=2, position=(0,0))

    """
    Tests:
    -pick up <load
    -pick up >= load, not close enough

    -drop <load
    -drop >=load, not close enough
        -all NOT in zone
        -partial amt in zone
    -drop after dropped

    -pick up after dropped
    """

    def test_init(self):
        p=self.init()
        self.assertEqual(p.load, 3)

    def test_pick_and_drop(self):
        p=self.init()
        agents=[AbsAgent(position=(1,1)) for _ in range(p.load)]
        d=Deposit(position=(0,0), radius=0)
        self.assertFalse(p.deposited)

        # --------------
        # drop before picked - should fail
        # TODO
        self.assertFalse(p.deposited)
        #drop <load & not even been picked
        success=False
        for a in agents:
            self.assertFalse(p.grabbed)
            self.assertFalse(p.deposited)
            # not successful since it hasn't been picked
            self.assertFalse(success)
            success,info=p.drop(a, d)
        #drop >=load, not picked up
        self.assertFalse(p.grabbed)
        self.assertFalse(p.deposited)

        # ---------------
        #pick up <load, not close enough
        p.set_radius(1) # distance is sqrt(2)>1
        # initially, 0,1,2 agents -> not enough
        for a in agents:
            self.assertFalse(p.grabbed)
            p.pick(a)
        #pick up >= load, not close enough
        self.assertFalse(p.grabbed)

        # ---------------
        #pick up >= load, close enough
        p.set_radius(2)
        # initially, 0,1,2 agents -> not enough
        for a in agents:
            self.assertFalse(p.grabbed)
            p.pick(a)
        #pick up >= load, close enough
        self.assertTrue(p.grabbed)

        # ---------------
        # check not deposited
        self.assertFalse(p.deposited)
        #drop <load & ALL not close enough
        success=False
        for a in agents:
            self.assertTrue(p.grabbed)
            self.assertFalse(p.deposited)
            # not successful since there aren't enough & not close enough
            self.assertFalse(success)
            success,info=p.drop(a, d)
        #drop >=load, not close enough
        self.assertTrue(p.grabbed)
        self.assertFalse(p.deposited)

        # --------------
        # DROP - only a couple close enough
        d.set_radius(1) # set not close enough for ANY, as distance is sqrt(2)<2
        # make 2 of them close enough (not enough critical mass)
        for i in range(2):
            agents[i].set_position(0,0)
        # NOT DEPOSITED
        self.assertFalse(p.deposited)
        #drop <load & ONE is not close enough
        success=False
        for a in agents:
            self.assertTrue(p.grabbed)
            self.assertFalse(p.deposited)
            # not successful since there aren't enough & ONE is not close enough
            self.assertFalse(success)
            success,info=p.drop(a, d)
        #drop >=load, ONE is not close enough
        self.assertTrue(p.grabbed)
        self.assertFalse(p.deposited)

        # -------------
        d.set_radius(2) # set close enough for ALL, as distance is sqrt(2)<2
        #drop <load, even if close enough
        success=False
        for a in agents:
            self.assertTrue(p.grabbed)
            self.assertFalse(p.deposited)
            self.assertFalse(success)
            success,info=p.drop(a, d)
        #drop >=load and close enough - success
        self.assertTrue(success)
        self.assertFalse(p.grabbed) # not grabbed anymore
        self.assertTrue(p.deposited)

        # -----------
        # Drop after dropped - should fail
        success=False
        for a in agents:
            self.assertFalse(p.grabbed) # not grabbed (by previous)
            self.assertTrue(p.deposited) # still deposited (by previous)
            self.assertFalse(success)
            success,info=p.drop(a, d)
        #drop >=load BUT it has already been dropped before - fails
        self.assertFalse(success)
        self.assertFalse(p.grabbed) # not grabbed (by previous)
        self.assertTrue(p.deposited) # still deposited (by previous)


    def test_step_couple(self):
        p=self.init()
        agents=[AbsAgent() for _ in range(p.load)]
        agents[0].set_position(1,1)

        for a in agents:
            p.pick(a)
        self.assertTrue(p.grabbed)
        
        p.step()

        self.assertEqual(p.get_position(), agents[0].get_position())

class AbsAgentTest(unittest.TestCase):
    def init(self):
        AbsAgent.INVENTORY_CAPACITY=2
        inv={'a' : 1, 'b' : 1}
        a=AbsAgent(inv)
        return a

    def test_init(self):
        p=self.init()
        self.assertEqual(p.occupied, 2)
        self.assertEqual(p.available, 0)
        self.assertTrue(p.full)

    def test_add(self):
        p=self.init()
        self.assertFalse(p.add_inventory('a'))
        self.assertEqual(p.used_space('a'), 1)
        self.assertEqual(p.used_space('b'), 1)

        p.inventory['A']=0
        self.assertTrue(p.add_inventory('a'))
        self.assertEqual(p.used_space('a'), 1)
        self.assertEqual(p.used_space('b'), 1)

        
        AbsAgent.set_capacity(4)

        # add 2
        self.assertTrue(p.add_inventory('a', 2))
        self.assertEqual(p.used_space('a'), 3)
        self.assertEqual(p.used_space('b'), 1)

        self.assertFalse(p.add_inventory('a', 2))
        self.assertEqual(p.used_space('a'), 3)
        self.assertEqual(p.used_space('b'), 1)

        # remove 2
        self.assertTrue(p.use_inventory('a', 2))
        self.assertEqual(p.used_space('a'), 1)
        self.assertEqual(p.used_space('b'), 1)

        self.assertFalse(p.use_inventory('a', 2))
        self.assertEqual(p.used_space('a'), 1)
        self.assertEqual(p.used_space('b'), 1)

        AbsAgent.set_capacity(2)

    def test_deposit_all(self):
        dct={'a' : 6, 'b' : 4, 'person' : 1}
        d=Deposit(capacity=11+2, storage=dct)
        p=self.init()
    
        self.assertTrue(p.deposit_all_inventory(d))

        self.assertEqual(p.used_space('a'), 0)
        self.assertEqual(p.used_space('b'), 0)

        self.assertEqual(d.space, 0)
        self.assertEqual(d.available('a'), 7)
        self.assertEqual(d.available('b'), 5)
        self.assertEqual(d.available('person'), 1)
        
        p._add_person(Person())
        self.assertTrue(p.deposit_all_inventory(d)) # true b/c trivial zero

    def test_changing_capacity(self):
        p=self.init()
        p.set_capacity(4)

        self.assertEqual(p.occupied, 2)
        self.assertEqual(p.available, 2)
        self.assertFalse(p.full)

        self.assertTrue(p.add_inventory('a'))
        self.assertTrue(p.add_inventory('b'))

        # back to 2
        p.set_capacity(2)

    def test_clear(self):
        p=self.init()

        p.inventory['PERSON']=1

        self.assertTrue(p.clear_inventory(including_person=False))
        self.assertEqual(p.used_space('a'), 0)
        self.assertEqual(p.used_space('b'), 0)
        self.assertTrue(p.has_person)
        self.assertFalse(p.full)


    def test_use(self):
        p=self.init()
        self.assertTrue(p.use_inventory('a'))
        self.assertFalse(p.use_inventory('a'))


class FieldTest(unittest.TestCase):
    
    def setUp(self):
        self.params=Coordinate.get_params()
        Coordinate.set_params(height=15,width=15)

    def tearDown(self):
        Coordinate.set_params(**self.params)

    def initfire(self, initial_pos, fire_type):
        flammables=[]
        idx=0
        xx,yy=initial_pos
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                xp,yp=xx+x, yy+y
                fl=Flammable(fire_type, position=(xp,yp))
                flammables.append(fl)

                if x==1 and y==1:
                    idx=x*3+y

        ctr=flammables.pop(idx)
        flammables.append(ctr) # add fire center at the end
        ctr.light()
        flammables[0].set_name("jamison flammable")
        fire=Fire(flammables, name="jamison fire", position=initial_pos)
        fire.make_potent()
        return fire

    def init(self, n_agents=1, fire_type='a'):
        agents=[AbsAgent(position=(1,1)) for _ in range(n_agents)]
        agents[0].set_name('jamison himself')

        person=Person(position=(4,4)) # load is 2
        # make so that one step in direction of person makes them visible
        # currently they're ~4.24 units away, +(1,1) would be ~2.83
        person.set_radius(3)
        persons=[person]

        reservoir=Reservoir(fire_type, 10, position=(14,14))
        reservoirs=[reservoir]

        deposit=Deposit(capacity=10, position=(13,13), radius=20)
        deposits=[deposit]

        fire=self.initfire((7,7), fire_type)
        fires=[fire]

        kwargs={
            'agents' : agents,
            'persons' : persons,
            'reservoirs' : reservoirs,
            'deposits' : deposits,
            'fires' : fires,
        }

        f=Field(**kwargs)
        return f

    def render(self, field):
        render(field.all_objects(expand=True, with_memory=True))

    def test_render(self):
        f=self.init(fire_type='b')
        # self.render(f)

    def test_names(self):
        f=self.init(1)
        # fire already has name, and flammable has name
        # one agent called "jamison himself"
        # test w/ expand=True, dct=True
        nm="jamison himself"
        a=f.get('agents', 0)
        self.assertEqual(f.name_get(nm), a)

        nm="jamison fire"
        fr=f.get('fires',0)
        self.assertEqual(f.name_get(nm), fr)

        nm="jamison flammable"
        fl=fr.all_objects()[0]
        self.assertEqual(f.name_get(nm), fl)

        self.assertEqual(f.name_get("no name here"), None)

        # remember that these functions are cached, edit before calling (once)
        nms=f.all_names(expand=True, dct=False)
        self.assertEqual(len(nms), 3) # currently fire+flammable+agent 0

    
    def test_init(self):
        f=self.init(1)
        # amount of objects including flammable
        self.assertEqual(len(f.all_objects(expand=True, with_memory=False)), (4+9))
        # amount of objects not including flammable
        self.assertEqual(len(f.all_objects(expand=False, with_memory=False)), (4+1))

        # amount of objects including flammable+fire
        self.assertEqual(len(f.all_objects(expand=True, with_memory=True)), (4+9+1))
        # amount of objects including flammable w/o fire (same as first)
        self.assertEqual(len(f.all_objects(expand=True, with_memory=False)), (4+9))

    def test_step(self):
        f=self.init()

        N=((Flammable.L_TO_M+1)*2+(Flammable.L_TO_M+Flammable.M_TO_H)+1)*Fire.STEP_EVERY
        for _ in range(N):
            f.step()

        # fire should be spread everywhere at max intensity
        for fl in f.get('fires',0).flammables:
            self.assertTrue(fl.on_fire())
            self.assertEqual(fl.intensity,Intensity.HIGH)

    def test_local_observation(self):
        f=self.init(n_agents=2)
        
        # test local partial observation
        a0=f.get('agents', 0)
        # render(f.all_objects(expand=True))
        amt_around=lambda lobs : sum([len(p) for p in lobs.values()])
        readable=lambda lobs : (dict([(k,list(map(lambda x : x.class_name(), v))) for k,v in lobs.items()]))

        # too far away from fire to see anything - but can see other agent (they're at same position)
        lobs=f.local_partial_observation(0)
        self.assertEqual(amt_around(lobs), 1)

        # right in the middle (surrounded)
        a0.set_position(7,7) # now it can't see the other agent locally ((7,7) vs (1,1))
        lobs=f.local_partial_observation(0)
        self.assertEqual(amt_around(lobs), 9)
        self.assertEqual(lobs[(0,0)][0].class_name(), "Flammable")

        # in the right corner (can only see 4)
        a0.set_position(7+1,7+1)
        lobs=f.local_partial_observation(0)
        self.assertEqual(amt_around(lobs), 4)
        visible_set=set([(0,0), (-1,0), (0,-1), (-1,-1)])
        non_trivial_keys=set([k for k,v in lobs.items() if len(v)>0])
        self.assertEqual(non_trivial_keys, visible_set)

        # in the top edge (can only see 6)
        a0.set_position(7,7+1)
        lobs=f.local_partial_observation(0)
        self.assertEqual(amt_around(lobs), 6)
        visible_set=set([(0,0), (-1,0), (0,-1), (1,0), (-1,-1), (1,-1)])
        non_trivial_keys=set([k for k,v in lobs.items() if len(v)>0])
        self.assertEqual(non_trivial_keys, visible_set)

        # print(readable(lobs))

    def test_partial_w_expand(self):
        # "jamison flammable"
        f=self.init(n_agents=2)

        obs=f.partial_observation(0,expand=True)
        self.assertTrue('Flammable' in [o.class_name() for o in obs])
        for o in obs:
            if o.class_name()=='Flammable':
                self.assertEqual(o.get_name(), "jamison flammable")

    def test_visibility(self):
        f=self.init(n_agents=2)

        obs=f.partial_observation(0)
        # fire+agent_2+deposit_1+reservoir is visible by agent
        self.assertEqual(len(obs), 4)
        # must be fire+agent+deposit+reservoir
        self.assertTrue(isinstance(obs[0], Fire))
        self.assertTrue(isinstance(obs[1], Reservoir))
        self.assertTrue(isinstance(obs[2], Deposit))
        self.assertTrue(isinstance(obs[3], AbsAgent))

        # now move agent_1 closer to person (within radius)
        f.get('agents',0).change_position(1,1)
        obs=f.partial_observation(0)
        # fire+deposit_1+reservoir+person should be visible by agent - agent 2 disappeared since we moved away
        self.assertEqual(len(obs), 4)
        # must be fire+agent
        self.assertTrue(isinstance(obs[0], Fire))
        self.assertTrue(isinstance(obs[1], Reservoir))
        self.assertTrue(isinstance(obs[2], Deposit))
        # self.assertTrue(isinstance(obs[3], AbsAgent))
        self.assertTrue(isinstance(obs[3], Person))
        
        # agent 2 should remain the same
        obs=f.partial_observation(1)
        # fire+agent_1+deposit_1+reservoir is visible by agent - symmetrically, agent 1 disappeared as well
        # AND person visible by agent, since if 1-agent sees it (within radius of person),
        # that person becomes visible to all
        self.assertEqual(len(obs), 4)
        # must be fire+agent+deposit+reservoir
        self.assertTrue(isinstance(obs[0], Fire))
        self.assertTrue(isinstance(obs[1], Reservoir))
        self.assertTrue(isinstance(obs[2], Deposit))
        # self.assertTrue(isinstance(obs[3], AbsAgent))
        self.assertTrue(isinstance(obs[3], Person))

        # Test visibility when person is picked up and deposited (should be none!)
        p=f.get('persons',0)
        d=f.get('deposits',0)
        # move agent 2 within radius of person
        f.get('agents',1).change_position(1,1)

        # pick up
        for a in f.get('agents'):
            self.assertFalse(p.grabbed)
            p.pick(a)
        self.assertTrue(p.grabbed)

        # drop
        for a in f.get('agents'):
            success,info=p.drop(a, d)
        self.assertTrue(success) # has been dropped
        self.assertFalse(p.grabbed) # not grabbed anymore
        self.assertTrue(p.deposited) # deposited properly

        # agent 1 shouldn't see person
        obs=f.partial_observation(0)

        self.assertTrue(p.deposited) # still deposited properly

        # fire+agent_1+deposit_1+reservoir is visible by agent - agent back in visibility radius
        self.assertEqual(len(obs), 4)
        # must be fire+agent+deposit+reservoir
        self.assertTrue(isinstance(obs[0], Fire))
        self.assertTrue(isinstance(obs[1], Reservoir))
        self.assertTrue(isinstance(obs[2], Deposit))
        self.assertTrue(isinstance(obs[3], AbsAgent))

        # print([o.class_name() for o in obs])

    def test_id(self):
        f=self.init(n_agents=2)
        # test that we can access by id
        fr=f.get('fires',0)
        self.assertEqual(f.id_get(fr.id),fr)

class GPSTest(unittest.TestCase):
    def setUp(self):
        self.params=Coordinate.get_params()
        Coordinate.set_params(height=20,width=20)

    def tearDown(self):
        Coordinate.set_params(**self.params)

    def test_update_track(self):
        previous_pos=Coordinate(0,0)
        o=Person(position=previous_pos.get())

        self.assertTrue(o.id in GPS.at(o.position))

        o.set_position(1,1)
        self.assertTrue(o.id in GPS.at(o.position))
        self.assertFalse(o.id in GPS.at(previous_pos))

        o2=Person(position=(2,2))
        d=GPS.near(o2.position, math.sqrt(2))
        l=functools.reduce(lambda l1,l2 : l1+l2, list(d.values()))
        self.assertTrue(o.id in l)

        
class ControllerTest(unittest.TestCase):
        
    def init(self, num_agents):
        agent_names=['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Finn'][:num_agents]
        params=Controller.pg_params(agent_names=agent_names)
        c=Controller(procedural_generation_parameters=params, seed=42)
        
        # print(c.get_inventory(0))
        # this is so illegal...
        # render(c.backend.engine.objects)

        """
        'NavigateTo' : to_target_id (location),
        'Move' : direction, # TODO
        'Explore', # TODO
        'Carry' : from_target_id (person),
        'DropOff' : from_target_id (person), to_target_id (deposit),
        'GetSupply' : from_target_id (deposit or reservoir), supply_type (type)
        'UseSupply' : from_target_id (fire), supply_type (type)
        'StoreSupply' : to_target_id (deposit), 
        'ClearInventory',
        """
        # names: ReservoirUtah, ReservoirYork, DepositFacility, CaldorFire, GreatFire, LostPersonTimmy, Alice, Bob, Charlie, David, Emma, Finn
        # w/ 2 regions each {fire_name}_region_{index+1}
        return c
    
    def test_procgen(self):
        c=self.init(num_agents=2)

    def render(self, c):
        render(c.backend.engine.objects)

    def test_store_supply_actions(self):
        n_agents=6
        agent_names=['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Finn']
        # names: ReservoirUtah, ReservoirYork, DepositFacility, CaldorFire, GreatFire, LostPersonTimmy, Alice, Bob, Charlie, David, Emma, Finn
        # Utah is type 'a', York is type 'b'
        # CaldorFire is type 'a', GreatFire is type 'b'

        c=self.init(num_agents=n_agents)
        # self.render(c)

        # --- all navigate to reservoirs & GetSupply ---
        for _ in range(2):
            for agent_idx in range(n_agents):

                # clear inventory
                event=c.step(action='ClearInventory', agent_idx=agent_idx)
                self.assertTrue(event['success'])
                # agent get all of them back
                self.assertEqual(c.get_inventory(agent_idx)['Sand'],0)
                self.assertEqual(c.get_inventory(agent_idx)['Water'],0)

                # -- get supplies --
                # --- utah
                reservoir_id=c.get_id('ReservoirUtah')

                # navigate to reservoir
                event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=reservoir_id)
                self.assertTrue(event['success'])

                # get supply from reservoir
                event=c.step(action='GetSupply', agent_idx=agent_idx, from_target_id=reservoir_id)
                self.assertTrue(event['success'])
                self.assertEqual(c.get_inventory(agent_idx)['Sand'],1)

                # ---- york
                reservoir_id=c.get_id('ReservoirYork')

                # navigate to reservoir
                event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=reservoir_id)
                self.assertTrue(event['success'])

                # get supply from reservoir
                event=c.step(action='GetSupply', agent_idx=agent_idx, from_target_id=reservoir_id)
                self.assertTrue(event['success'])
                self.assertEqual(c.get_inventory(agent_idx)['Water'],1)


        # self.render(c)

        # --- all navigate to deposit & StoreSupply ---
        for agent_idx in range(n_agents):
            deposit_id=c.get_id('DepositFacility')
            
            # TODO: this one is new, test
            # put supply in deposit
            event=c.step(action='StoreSupply', agent_idx=agent_idx, to_target_id=deposit_id)
            self.assertFalse(event['success']) # not close enough to supply
            self.assertEqual(event['error_type'], "not_interactable") # not close enough to supply

            # navigate to deposit
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=deposit_id)
            self.assertTrue(event['success'])

            # put supply in deposit
            event=c.step(action='StoreSupply', agent_idx=agent_idx, to_target_id=deposit_id)
            self.assertTrue(event['success'])

            # agent drops all
            self.assertEqual(c.get_inventory(agent_idx, 'Sand'),0)
            self.assertEqual(c.get_inventory(agent_idx, 'Water'),0)

        # assert inventory of deposit has all
        deposit_d=list(filter(lambda x : x['name']=='DepositFacility', event['global_obs']))[0]
        self.assertEqual(deposit_d['inventory'].get('Sand'),n_agents)
        self.assertEqual(deposit_d['inventory'].get('Water'),n_agents)
            
        # --- while still near deposit, all navigate to location
        # this part of the test only works if the size of inventory is 2
        if AbsAgent.INVENTORY_CAPACITY==2:
            # -- get supply --
            for agent_idx in range(n_agents):
                # NOTE: since we'll fill inventory with this, you can't get the supply back individually
                # so we will alternate between them
                # self.render(c)
                if agent_idx%2==0:
                    # pickup supply from deposit
                    event=c.step(action='GetSupply', agent_idx=agent_idx, from_target_id=deposit_id, supply_type='Sand')
                    self.assertTrue(event['success'])
                    # agent get all of them back
                    self.assertEqual(c.get_inventory(agent_idx)['Sand'],AbsAgent.INVENTORY_CAPACITY)
                else:
                    # pickup supply from deposit
                    event=c.step(action='GetSupply', agent_idx=agent_idx, from_target_id=deposit_id, supply_type='Water')
                    self.assertTrue(event['success'])
                    # agent get all of them back
                    self.assertEqual(c.get_inventory(agent_idx)['Water'],AbsAgent.INVENTORY_CAPACITY)

        # self.render(c)

        # --- all navigate to fires & UseSupply ---
        for agent_idx in range(n_agents):
            if agent_idx%2==0:
                # --- utah
                fire_id=c.get_id('CaldorFire')
                supply_type='Sand'
                # TODO: mapping from readable supply to non-readable

                # navigate to reservoir
                event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=fire_id)
                self.assertTrue(event['success'])

                # use supply from reservoir
                event=c.step(action='UseSupply', agent_idx=agent_idx, to_target_id=fire_id, supply_type=supply_type)
                self.assertTrue(event['success'])

            else:
                # ---- york
                fire_id=c.get_id('GreatFire')
                supply_type='Water'

                # navigate to reservoir
                event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=fire_id)
                self.assertTrue(event['success'])

                # use supply from reservoir
                event=c.step(action='UseSupply', agent_idx=agent_idx, to_target_id=fire_id, supply_type=supply_type)
                self.assertTrue(event['success'])
            # self.render(c)

        # self.render(c)

        # Make sure to check that if fire is here, so is its regions - which still is the case
        gobs=event['global_obs']
        gobs_types=list(map(lambda d : d['type'], gobs))
        has_fire='Fire' in gobs_types
        if has_fire: self.assertTrue('Flammable' in gobs_types)

        # print('global:')
        # pprint(event['global_obs'])
        # print('local:')
        # pprint(event['local_obs'])

    def test_fire_stepping(self):
            n_agents=6
            agent_names=['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Finn']
            # names: ReservoirUtah, ReservoirYork, DepositFacility, CaldorFire, GreatFire, LostPersonTimmy, Alice, Bob, Charlie, David, Emma, Finn
            # Utah is type 'a', York is type 'b'
            # CaldorFire is type 'a', GreatFire is type 'b'

            c=self.init(num_agents=n_agents)

            """
            print('fire stepping..............')
            fire=c.get('fires',0)
            flammables=fire.all_objects()
            print('on fire', [int(f.on_fire()) for f in flammables])
            print('intensity', [(f.intensity.value) for f in flammables])
            print('clocks', [int(f._clock) for f in flammables])
            print()
            print('fire stepping..............end')
            """

            # --- all navigate to reservoirs & do nothing (to get fire going) ---
            for _ in range(4):
                for agent_idx in range(n_agents):
                    # --- utah
                    reservoir_id=c.get_id('ReservoirUtah')

                    # navigate to reservoir
                    event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=reservoir_id)
                    self.assertTrue(event['success'])
                    # pprint(event['global_obs'])

                # self.render(c)

    # TODO: Test fire progression stepping (when doing UseSupply) function
    # TODO: is the average be none when extinguished
    def test_fire_supply_actions(self):
        n_agents=6
        agent_names=['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Finn']
        # names: ReservoirUtah, ReservoirYork, DepositFacility, CaldorFire, GreatFire, LostPersonTimmy, Alice, Bob, Charlie, David, Emma, Finn
        # Utah is type 'a', York is type 'b'
        # CaldorFire is type 'a', GreatFire is type 'b'

        c=self.init(num_agents=n_agents)
        # render(c.backend.engine.objects)

        # --- all navigate to reservoirs & GetSupply ---
        for agent_idx in range(n_agents):
            # --- utah
            reservoir_id=c.get_id('ReservoirUtah')

            # get supply from reservoir - but before you're there!  (fails)
            event=c.step(action='GetSupply', agent_idx=agent_idx, from_target_id=reservoir_id)
            self.assertFalse(event['success'])
            self.assertEqual(c.get_inventory(agent_idx)['Sand'],0)

            # navigate to reservoir
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=reservoir_id)
            self.assertTrue(event['success'])

            # get supply from reservoir
            event=c.step(action='GetSupply', agent_idx=agent_idx, from_target_id=reservoir_id)
            self.assertTrue(event['success'])
            self.assertEqual(c.get_inventory(agent_idx)['Sand'],1)

            # ---- york
            reservoir_id=c.get_id('ReservoirYork')

            # get supply from reservoir - but before you're there!  (fails)
            event=c.step(action='GetSupply', agent_idx=agent_idx, from_target_id=reservoir_id)
            self.assertFalse(event['success'])
            self.assertEqual(c.get_inventory(agent_idx)['Water'],0)

            # navigate to reservoir
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=reservoir_id)
            self.assertTrue(event['success'])

            # get supply from reservoir
            event=c.step(action='GetSupply', agent_idx=agent_idx, from_target_id=reservoir_id)
            self.assertTrue(event['success'])
            self.assertEqual(c.get_inventory(agent_idx)['Water'],1)

        # let the fire roar for a bit
        # --- all navigate to reservoirs & do nothing (to get fire going) ---
        # self.render(c)
        for _ in range(4):
            for agent_idx in range(n_agents):
                # --- utah
                reservoir_id=c.get_id('ReservoirUtah')

                # navigate to reservoir
                event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=reservoir_id)
                self.assertTrue(event['success'])
                # pprint(event['global_obs'])
            # self.render(c)

        # --- all navigate to fires & UseSupply ---
        for agent_idx in range(n_agents):
            # --- utah
            fire_id=c.get_id('CaldorFire')
            supply_type='Sand'
            # TODO: mapping from readable supply to non-readable

            # navigate to reservoir
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=fire_id)
            self.assertTrue(event['success'])

            # get supply from reservoir
            event=c.step(action='UseSupply', agent_idx=agent_idx, to_target_id=fire_id, supply_type=supply_type)
            self.assertTrue(event['success'])
            # self.render(c)

            # ---- york
            fire_id=c.get_id('GreatFire')
            supply_type='Water'

            # navigate to reservoir
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=fire_id)
            self.assertTrue(event['success'])
            # self.render(c)

            # get supply from reservoir
            event=c.step(action='UseSupply', agent_idx=agent_idx, to_target_id=fire_id, supply_type=supply_type)
            self.assertTrue(event['success'])

            # print('inventory:', c.get_inventory(agent_idx))

        # check that the fire has been tamed


    def test_carry_dropoff(self):
        n_agents=6
        agent_names=['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Finn']
        # names: ReservoirUtah, ReservoirYork, DepositFacility, CaldorFire, GreatFire, LostPersonTimmy, Alice, Bob, Charlie, David, Emma, Finn

        c=self.init(num_agents=n_agents)
        # render(c.backend.engine.objects)

        # --- all carry ---
        successes=[]
        for agent_idx in range(n_agents):
            person_id=c.get_id('LostPersonTimmy')

            if agent_idx==0:
                # navigate to person - failure (not visible for first one yet)
                event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=person_id)
                self.assertFalse(event['success'])
                self.assertEqual(event['error_type'], "not_visible")

                # cheat - direct jump to it
                person, agent=c.id_get(person_id), c.get('agents', agent_idx)
                eps_radius=person.get_radius()
                success,info=c.backend.navigate(agent, person.position, eps=eps_radius)
                self.assertTrue(success)


            # navigate to person
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=person_id)
            self.assertTrue(event['success'])

            # carry person (try w/ 1)
            event=c.step(action='Carry', agent_idx=agent_idx, from_target_id=person_id)
            # NOTE: don't assert any success for one since it'll need all
            successes.append(event['success'])

        # they were able to carry!
        self.assertTrue(any(successes))

        # --- all navigate to deposit & (bad) dropoff ---
        for agent_idx in range(n_agents):
            deposit_id=c.get_id('DepositFacility')

            # navigate to deposit
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=deposit_id)
            self.assertTrue(event['success'])

            # try to drop person (try w/ 1)
            event=c.step(action='DropOff', agent_idx=agent_idx, from_target_id=person_id, to_target_id=deposit_id)
            # render(c.backend.engine.objects)

        # --- (good) dropoff ---
        successes=[]
        for agent_idx in range(n_agents):
            deposit_id=c.get_id('DepositFacility')

            # try to drop person (try w/ 1)
            event=c.step(action='DropOff', agent_idx=agent_idx, from_target_id=person_id, to_target_id=deposit_id)
            
            # dropping should be unsuccessful for all
            # NOTE: not even the last one is successful since they all have to be close AND THEN you can begin to deposit
            # NOTE: since the last one above dropoff after the others were already there, it should be successful @ the one before last
            
            # render(c.backend.engine.objects)
            successes.append(event['success'])
        self.assertTrue(any(successes))

    def test_move(self):
        n_agents=6
        agent_names=['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Finn']
        # names: ReservoirUtah, ReservoirYork, DepositFacility, CaldorFire, GreatFire, LostPersonTimmy, Alice, Bob, Charlie, David, Emma, Finn

        c=self.init(num_agents=n_agents)
        # render(c.backend.engine.objects)

        for agent_idx in range(n_agents):
            # go to deposit
            direction='Right'
            event=c.step(action='Move', agent_idx=agent_idx, to_target_id=direction)
            self.assertTrue(event['success'])

            # move a ton in direction (at least 30)
            for _ in range(max(Coordinate.HEIGHT, Coordinate.WIDTH)):
                event=c.step(action='Move', agent_idx=agent_idx, to_target_id=direction)

            # it has hit a wall so it must fail
            event=c.step(action='Move', agent_idx=agent_idx, to_target_id=direction)
            self.assertFalse(event['success'])

            # render(c.backend.engine.objects)

    def test_navigation(self):
        n_agents=6
        agent_names=['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Finn']
        c=self.init(num_agents=n_agents)
        # render(c.backend.engine.objects)
        # names: ReservoirUtah, ReservoirYork, DepositFacility, CaldorFire, GreatFire, LostPersonTimmy, Alice, Bob, Charlie, David, Emma, Finn

        for agent_idx in range(n_agents):
            # go to deposit
            _id=c.get_id('DepositFacility')
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=_id)
            self.assertTrue(event['success'])

            # go to fire one
            _id=c.get_id('CaldorFire')
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=_id)
            self.assertTrue(event['success'])

            # go to fire one region 1
            _id=c.get_id('CaldorFire_Region_1')
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=_id)
            self.assertTrue(event['success'])

            # go to fire one region 2
            _id=c.get_id('CaldorFire_Region_2')
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=_id)
            self.assertTrue(event['success'])

            # go to fire two region 1
            _id=c.get_id('GreatFire_Region_1')
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=_id)
            self.assertTrue(event['success'])
            
            # go to fire two region 2
            _id=c.get_id('GreatFire_Region_2')
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=_id)
            self.assertTrue(event['success'])

            # go to ReservoirUtah
            _id=c.get_id('ReservoirUtah')
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=_id)
            self.assertTrue(event['success'])

            # go to ReservoirYork
            _id=c.get_id('ReservoirYork')
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=_id)
            self.assertTrue(event['success'])

            # go to LostPersonTimmy - false, shouldn't be able to see it
            _id=c.get_id('LostPersonTimmy')
            event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=_id)
            self.assertFalse(event['success'])

            for i in range(n_agents):
                if i>=agent_idx: continue # can only go to agents that are around the deposit (close enough)

                # go to agent i
                # should fail - can't navigate to another agent (if you could, it would be visible)
                to_agent_id=c.get_id(agent_names[i])
                me,them=c.get('agents',agent_idx),c.get('agents',i)

                event=c.step(action='NavigateTo', agent_idx=agent_idx, to_target_id=to_agent_id)
                self.assertTrue(me.id not in [d['id'] for d in event['global_obs']]) # not self-referential
                self.assertTrue('AbsAgent' in [d['type'] for d in event['global_obs']])
                self.assertTrue(event['success']) # we can see this agent (by the triangular restriction above)



if __name__ == "__main__":
    # TODO: add unit testing for all of these
    unittest.main()

