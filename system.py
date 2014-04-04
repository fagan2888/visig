#!/usr/bin/env python
from visig import OrderedIndexedDict
from functools import partial

# functions:
#   - registers and manages sig proc systems
#   - generates algorithms by chaining systems
class AlgoMng(object):
    '''
    an algorithm can be applied against at most one input signal
    a signal can have multiple algorithms producing computations on it

    make a request to this manager for an algorithm and apply it to an input
    '''
    def __init__(self, signal, frame_len=512):
        self._sys_cache = {}
        self.signal = signal
        self._frame_len = frame_len
        self._algos = OrderedIndexedDict()

    def register(self, system):
        '''register a system in the manager'''
        if isinstance(system, System):
            self._mcache[system.name] = system

class Algorithm(object):
    '''
    This class organizes a collection of signal processing systems into a directed
    graph to produce an algorithm. An algorithm can reports feature computations
    on the input signal and may produce an ouput signal
    '''
    def __init__(self):
        pass

class System(object):
    '''
    A class which implements a signal processing system T such that
    in y[n] = T{x[n]}.
    Use this class as a module which may be serialized/parallelized
    with other modules in order to build a signal processing chain.
    '''
    def __init__(self, frame_len, parent_sys=None, child_sys=None):
        self.out_frame = Signal(np.zeros(frame_len))
        # only expect the input frame on each iteration
        self.process = functools.partial(self._process, out_frame=self.out_frame)
        self._prev = parent_sys
        self._next = child_sys
        self.config_axes()

    def __call__(self, in_array):
        '''use call as the executor'''
           ouputs = self.process(in_array)
           return outputs,

   def export_feature(self, name, array):
       '''export a computation to the system'''
       pass

   def import_feature(self, name):
       ''' import data from a feature computation
       the feature must be computed by an upstream processor '''
       pass

    def configure_axes(self, fig):
        '''generate axes configuration for this system
        configuration steps might include creating new axes'''
        pass

    def process(self, in_frame, out_frame, *args):
        raise NotImplementedError("this routine must be implemented in the child class")

class Periodogram(System):
    def process(self, in_frame, out_frame):
        out_frame[:] = abs(fft(array))

class QuickRollingMean(System):
    # init settings
    def initialize(self, *args, **kwargs):
        transparent = True
        self.Tw = 1.0    # window len in seconds

@sparams(Tw=1.0)
def rollingmean(in_array, out_array, *args):
    # M = getattr(self, Tw, len(in_frame))
    M = rollingmean.Tw * len(in_frame))
    a = 1/M
    b = 1-a
    ax = np.zeros(len(vector))
    for n,v in enumerate(vector):
        ax[n] = a*v + b*(ax[n-1])
    return ax

# can we build a mechanism which composes/cascades systems?
class Agc(System):
    G  = -20.0  # dBFS

    def process(self, in_frame, out_frame, *args):
        # mx = self.get_feature('QuickRollingMean')
        ax = mx(abs(in_frame), Tw*Fs)
        gains = tg/ax
        return (frame * gains, gains, ax)
        return agc(array)

def agc(vector, tg=g):
    '''tg can be scalar or vector'''

# should be called after each sparam adjustment
def setattr_with_sparams(f_inst, name, value):
    pass

# 'stateful parameters' decorator
def sparams(**kwargs):
    def decorate(f):
        f.sparams = object()
        for n,v in kwargs.items():
            # setattr(f.sparams, n, v)
            f.sparams[n]
        f.__setattr__ = setattr_with_sparams
        return f
    return decorate
