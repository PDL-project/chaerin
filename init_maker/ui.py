import pygame
from init_maker_easy.utils import *
from collections import defaultdict
from init_maker_easy.viz_utils import *

# code (slider, label, eventhandler) inspired by https://github.com/m1chaelwilliams/Pygame-Tutorials/blob/main/ui/ui.py
UNSELECTED = "red"
SELECTED = "white"
BUTTONSTATES = {
    True:SELECTED,
    False:UNSELECTED
}

class EventHandler:
    def __init__(self):
        self.event = None

    @staticmethod
    def get():
        return pygame.event.get()

    def set_event(self, event):
        self.event = event

    def quit(self):
        return self.event.type == pygame.QUIT

    def click(self):
        return self.event.type == pygame.MOUSEBUTTONUP

    def key_press(self):
        return self.event.type == pygame.KEYDOWN

    def key(self, key):
        if key.lower() == 'space':
            return self.event.key == pygame.K_SPACE
        elif key.isalnum():
            key_event = getattr(pygame, f"K_{key}") 
            return self.event.key == key_event
        else:
            raise ValueError(f"key {key} not valid to check")

class UI:
    @staticmethod
    def init():
        UI.font = pygame.font.Font(None, 24)
        UI.sfont = pygame.font.Font(None, 16)
        UI.lfont = pygame.font.Font(None, 32)
        UI.xlfont = pygame.font.Font(None, 48)

        UI.rescale_ratio = .75

        # initialize background (no blit-ing of image)
        UI._background()
        UI.screen = pygame.display.set_mode((UI.window_width, UI.window_height))

        UI.slider_size = (300, 15)
        UI.slider_pos = (10+UI.slider_size[0]//2, (UI.window_height//8)+UI.slider_size[1]//2)
        def shift(pos, n=1, fonth=32):
            w,h=pos
            ws,hs=UI.slider_size
            h+=(hs+fonth)*n
            return (w,h)

        UI.sliders = [
            Slider(pos=UI.slider_pos, size=UI.slider_size, initial_val=0.5, min=-1, max=1, name="x: "),
            Slider(pos=shift(UI.slider_pos,n=1), size=UI.slider_size, initial_val=0.5, min=-1, max=1, name="y: "),
            Slider(pos=shift(UI.slider_pos,n=2), size=UI.slider_size, initial_val=1.2/2, min=-1, max=1, name="height: "),
        ]

        # render background image
        UI.background(recalc=False)

    @staticmethod
    def _sliders():
        mouse_pos = pygame.mouse.get_pos()
        mouse = pygame.mouse.get_pressed()
        for slider in UI.sliders:
            if slider.container_rect.collidepoint(mouse_pos):
                if mouse[0]:
                    slider.grabbed = True
            if not mouse[0]:
                slider.grabbed = False
            if slider.button_rect.collidepoint(mouse_pos):  
                slider.hover()
            if slider.grabbed:
                slider.move_slider(mouse_pos)
                slider.hover()
            else:
                slider.hovered = False
            slider.render()
            slider.display_value()

    @staticmethod
    def background(recalc=True, crop=None):
        # wrap _background call w/ blit
        if recalc:
            UI._background(crop=crop)
        UI.blit(UI.bg_image)


    @staticmethod
    def _background_params():
        UI.window_width = UI.rect.w
        UI.window_height = UI.rect.h
        UI.center = (UI.window_width//2, UI.window_height//2)
        UI.half_width = UI.center[0]
        UI.half_height = UI.center[1]

        UI.fonts = {
            'sm':UI.sfont,
            'm':UI.font,
            'l':UI.lfont,
            'xl':UI.xlfont
        }


    @staticmethod
    def _background(crop=None):
        # updates background
        # crop -> output of get_size(env) or None

        # loading the background
        map_data = get_agent_map_data(UI.env)
        scene_topview=map_data["frame"]
        bg_image = pygame.image.frombuffer(scene_topview.tostring(), scene_topview.shape[1::-1], "RGB")
        rect = bg_image.get_rect()

        # rescale to fit
        imgh, imgw = rect.h, rect.w
        bg_image = pygame.transform.scale(bg_image, (imgh * UI.rescale_ratio, imgw * UI.rescale_ratio))
        rect = bg_image.get_rect()

        if crop is not None:
            dx,dy=crop['x'],crop['z']
            # assuming that only side has to be cropped - crop the non-usable surface of floorplan
            sidecrop=round(rect.w - (rect.h/dy)*dx)
            bg_image=bg_image.subsurface((math.floor(sidecrop/2), 0, rect.w - sidecrop, rect.h))
            rect = bg_image.get_rect()

        UI.bg_image, UI.rect, UI.map_data = bg_image, rect, map_data
        UI._background_params()

    @staticmethod
    def blit(img, dest=(0,0)):
        UI.screen.blit(img, dest)

    @staticmethod
    def text(text, color=(255,255,255)):
        return UI.font.render(text, True, color)

    def shift_text(top_text, n=1):
        rect_text = top_text.get_rect()
        _,wrd_height=top_text.get_size()
        return wrd_height*n

    @staticmethod
    def refresh(sel_obj_name, error=None):
        # add background image (no recalculation)
        UI.background(recalc=False)

        # render top text
        text_obj = "NONE" if sel_obj_name is None else sel_obj_name
        text_surface_l1 = UI.text(f"sel object: {text_obj.upper()};")
        text_surface_l2 = UI.text(f"click a or d (left/right) to toggle objects;      click m to move;")

        UI.blit(text_surface_l1, dest=(0,0))
        UI.blit(text_surface_l2, dest=(0,UI.shift_text(text_surface_l1, n=1)))

        if error == -1:
            emessage = f"movement successful!"
            text_surface_l3 = UI.text(emessage)
            UI.blit(text_surface_l3, dest=(0,UI.shift_text(text_surface_l2, n=2)))
        elif error is not None:
            emessage = f"movement error, see terminal! (likely out-of-bounds or interfering)"
            text_surface_l3 = UI.text(emessage)
            UI.blit(text_surface_l3, dest=(0,UI.shift_text(text_surface_l2, n=2)))

        # render sliders
        UI._sliders()

        # render
        pygame.display.flip()

class Menu:
    def __init__(self, inith) -> None:
        # initialize
        UI.init()

        self.ehandler = EventHandler()
        # states
        # self.states=["toggle (select)", "move"]
        # self.state_index=0

        # refresh (initial render w/ text)
        UI.refresh(None, error=None)

        self.inith = inith

    def run(self):
        out = self.inith.prerun_actions()
        UI.background(recalc=True)

        if out == -1: # out -> -1 means no movement to be taken, exit
            self.inith.save()   # save on the way
            return

        running = True
        while running:
            out = self.loop()

            UI.refresh(self.inith.sel_obj_name, self.inith.error)
            if out == -1:
                running = False

        self.inith.output()
        # save pre-init to proper position
        self.inith.save()

    def loop(self):
        events = EventHandler.get()
        for event in events:
            # set current event for handler
            self.ehandler.set_event(event)

            # handle quit - send message above to stop while loop
            if self.ehandler.quit():
                return -1

            # handle MOUSEBUTTONUP
            if self.ehandler.click():
                pos = pygame.mouse.get_pos()
                pass

            # handle button press
            if self.ehandler.key_press():

                if self.ehandler.key('d'):
                    self.inith.shift_obj_idx(1)
                elif self.ehandler.key('a'):
                    self.inith.shift_obj_idx(-1)

                # if m move (as long as state and selection is correct)
                if self.ehandler.key('m') and self.inith.sel_obj_curr is not None:
                    xr, yr, h = [s.get_value() for s in UI.sliders]

                    # shift object by said amount
                    self.inith.shift_object([xr, h, yr])

                    # update bg image
                    UI.background(recalc=True)


class InitHandler:

    def __init__(self, metadata):
        # all objects in env
        self.all_objs = all_objects(UI.env)
        self.all_objs = list(filter(lambda x : x["pickupable"], self.all_objs))
        self.pos_tuple=lambda o : (o['position']['x'], o['position']['z'])

        # objects selected (mem + current)
        self.sel_obj_mem = set()
        self.sel_obj_idx = 0
        self.error = None # when moving object
        self.initial_objs_d = self.all_objs

        # data from json file
        self.metadata = metadata
        # edit actions to use id
        self.metadata["actions"]=self.actions_metadata_convert()

    def shift_obj_idx(self, val):
        self.sel_obj_idx=(self.sel_obj_idx+val)%len(self.all_objs)

    def shift_object(self, tpl):
        xr, h, yr = tpl
        to_position=InitHandler.to_position(self.sel_obj_curr['position'], InitHandler._abs_offset([xr, h, yr]))
        ev=UI.env.controller.step(
            action='PlaceObjectAtPoint',
            objectId=self.sel_obj_curr['objectId'],
            position=to_position
        )
        if ev.events[0].metadata["actionReturn"] is None:
            self.error = ev.events[0].metadata["errorMessage"]
            print("MOVE ERROR:\t", ev.events[0].metadata["errorMessage"])
        else:
            # add object to mem (for final tally)
            self.sel_obj_mem.add(self.sel_obj_curr['objectId'])
            self.error = -1

            # update objects and their position
            self.all_objs = all_objects(UI.env)
            self.all_objs = list(filter(lambda x : x["pickupable"], self.all_objs))
            print("Movement Successful!")

    def output(self):
        # after window is closed give summary to use in the pre-init function on python
        print("\n")
        s=" Summary of Object Movements "
        print("*"*len(s))
        print(s)
        print("*"*len(s))
        print()

        for obj_id in self.sel_obj_mem:
            obj_name=obj_id.split('|')[0]

            obj_old=get_object_from_id(self.initial_objs_d, obj_id)
            obj_new=get_object_from_id(self.all_objs, obj_id)

            initial_pos=obj_old['position']
            final_pos=obj_new['position']
            dict_diff = lambda di,df : dict([(k,df[k]-di[k]) for k in di.keys()])
            rel_diff_vector = rel_offset(UI.env, list(dict_diff(initial_pos, final_pos).values()))
            rel_diff_vector[1]+=.01 # correct z direction (add non-zero value to prevent clipping)
            print(f"OBJECT ID: {obj_id} (name: {obj_name}).")
            print("RELATIVE DIFF ARRAY:", rel_diff_vector)
            print("ABSOLUTE POSITION DICT:", final_pos)
            print("\n")

    def save(self):
        # save the pre-init in proper file!
        positions_d=self.final_id_position_dict()
        actions_d=self.metadata["actions"]
        fs=write_meta(place_objects=positions_d, actions=actions_d, forceAction=True, evalmode=False)
        # path -> task_dir/file_name as str
        path=self.metadata["task_folder"]+'/'+self.metadata["floorplan"]
        fn=wrap_meta_result(fs, docs=self.metadata["docs"], path=path)
        print(f"\nPre-Init has been created in '{fn}'!\n")

    def prerun_actions(self):
        positions_d = None
        actions_d = self.metadata["actions"]
        if not bool(actions_d): # dictionary is empty
            return 0

        eval_list=write_meta(place_objects=positions_d, actions=actions_d, forceAction=True, evalmode=True)
        controller=UI.env.controller
        for e in eval_list:
            eval(e) # run the initialization for that action-object in the env

        if self.metadata["no_movement"]:
            return -1
        return 0
 
    def final_id_position_dict(self):
        # get output of position - correct format for write_meta(.)
        id_pos=dict()
        for obj_id in self.sel_obj_mem:
            obj_new=get_object_from_id(self.all_objs, obj_id)
            position=obj_new["position"]
            id_pos[obj_id]=position
        return id_pos 

    def actions_metadata_convert(self):
        # convert values from words to id - correct format for write_meta(.)
        # metadata["actions"] -> {"action_name" : [objects (names NOT ids)...]}
        if self.metadata["actions"]=="none":
            return None

        d = defaultdict(list)
        for action, obj_list in self.metadata["actions"].items():
            for obj_name in obj_list:
                # give option to raise error w/ ignore=False (when would this be useful though?)
                d[action]+=get_id(UI.env, obj_name, ignore=True, exclusive=True) # output is list of all applicable objects (with said name)
        return d

    @staticmethod
    def _abs_offset(tpl):
        xr, h, yr = tpl
        return abs_offset(UI.env, [xr, h, yr])

    @staticmethod
    def to_position(init_pos, abs_offset):
        return m_offset_position(UI.env, init_pos, abs_offset)

    @property
    def sel_obj_curr(self):
        return self.all_objs[self.sel_obj_idx]

    @property
    def sel_obj_name(self):
        return None if self.sel_obj_curr is None else self.sel_obj_curr['objectId'].split('|')[0]


class Label:
    def __init__(self, font: str, content: str, pos: tuple, value = "blue", selected: bool = False) -> None:
        self.font = font
        self.selected = selected
        self.content = content

        self.value = value

        self.text = UI.fonts[self.font].render(content, True, BUTTONSTATES[self.selected], None)
        self.text_rect = self.text.get_rect(center = pos)

    def render(self):
        UI.blit(self.text, self.text_rect)
        

class Slider:
    def __init__(self, pos: tuple, size: tuple, initial_val: float, min: int, max: int, name : str) -> None:
        self.pos = pos
        self.size = size
        self.hovered = False
        self.grabbed = False

        self.slider_left_pos = self.pos[0] - (size[0]//2)
        self.slider_right_pos = self.pos[0] + (size[0]//2)
        self.slider_top_pos = self.pos[1] - (size[1]//2)

        self.min = min
        self.max = max
        self.initial_val = (self.slider_right_pos-self.slider_left_pos)*initial_val # <- percentage

        self.container_rect = pygame.Rect(self.slider_left_pos, self.slider_top_pos, self.size[0], self.size[1])
        self.button_rect = pygame.Rect(self.slider_left_pos + self.initial_val - 5, self.slider_top_pos, 10, self.size[1])

        # label
        self.name = name
        self.text = UI.fonts['m'].render(self.name+str(self.get_value()), True, "white", None)
        self.label_rect = self.text.get_rect(center = (self.pos[0], self.slider_top_pos - 15))

    def _round(self, value, default=False):
        if default:
            return int
        # round to nearest quarter-10th (.025)
        return int(value * 40) / 40

        
    def move_slider(self, mouse_pos):
        pos = mouse_pos[0]
        if pos < self.slider_left_pos:
            pos = self.slider_left_pos
        if pos > self.slider_right_pos:
            pos = self.slider_right_pos
        self.button_rect.centerx = pos

    def hover(self):
        self.hovered = True

    def render(self):
        pygame.draw.rect(UI.screen, "darkgray", self.container_rect)
        pygame.draw.rect(UI.screen, BUTTONSTATES[self.hovered], self.button_rect)

    def get_value(self):
        val_range = self.slider_right_pos - self.slider_left_pos - 1
        button_val = self.button_rect.centerx - self.slider_left_pos
        return self._round((button_val/val_range)*(self.max-self.min)+self.min)

    def display_value(self):
        self.text = UI.fonts['m'].render(self.name+str(self.get_value()), True, "white", None)
        UI.blit(self.text, self.label_rect)
