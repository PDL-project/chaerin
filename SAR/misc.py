import random
import warnings
import inspect
import random
import time
import functools
import re

def extract_number(s):
    l=re.findall(r'\d+',s)
    if len(l)==0:   return None
    return int(l[0])

# Useful Function
def join_conjunction(l, conj):
    if len(l)<2: return str(l[0])
    if len(l)==2: return f"{l[0]} {conj} {l[1]}"
    return ", ".join(l[:-1]) + f", {conj} {l[-1]}"

def read_enum(s):
    return str(s).split('.')[1].lower()

def set_seed(num):
    random.seed(num)

""" ------ wrapper/s ------ """
# from - https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically
def deprecated(comment=""):
    def _deprecated(func):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used."""
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn(f"Call to deprecated function {func.__name__}. {comment}",
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return func(*args, **kwargs)
        return new_func
    return _deprecated

def hash_dict(d):
    l=[]
    for k,v in d.items():
        if isinstance(v, list):
            v=tuple(v)
        l.append((k,v))
    return frozenset(l)
unhash_dict=lambda fs : dict(list(fs))

def hashable_with_cache(fn):
    @functools.cache
    def __fn(hashed_dct, **kwargs):
        return fn(unhash_dict(hashed_dct), **kwargs)
    def _fn(unhashed_dct, **kwargs):
        return __fn(hash_dict(unhashed_dct), **kwargs)
    return _fn

def clocked_cache(fn):
    @functools.lru_cache(maxsize=1)
    def __fn(self, clock, *args, **kwargs):
        return fn(self, *args, **kwargs)
    def _fn(self, *args, **kwargs):
        return __fn(self, self.clock, *args, **kwargs)
    return _fn

def timeit(f):
    def _time(*args, **kwargs):
        # time decorator
        a=time.time()
        output=f(*args, **kwargs)
        b=time.time()
        print(f"Function {f.__name__} took {b-a} seconds to run")
        return output
    return _time

""" --- fns ---- """

def tuple_add(t1, t2):
    t1l,t2l=len(t1),len(t2)
    lmax=max(t1l,t2l)
    # adds padding to max
    t1+=tuple(0 for _ in range(lmax-t1l))
    t2+=tuple(0 for _ in range(lmax-t2l))
    return tuple([t1v+t2v for t1v,t2v in zip(t1,t2)])


def clamp(v, mi, ma):
    return max(min(v, ma), mi)

def upper_keys(d):
    # upper() the key strings
    for k in list(d.keys()):
        d[k.upper()]=d.pop(k)

class Arg:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,str(k),v)
