#!/usr/bin/env python3
'''
Visig   - visualize and animate signals using mpl
'''
# TODO:
# - consider using sox conversion in this module so we can open
#   arbitrarly formatted audio files as numpy arrays (pysox?)
# - 'Signal' class which is delegated all data unpacking and metadata maintenance
# - animation for simple playback using a cursor on the chosen axes (add mixing?)
# - animation for rt spectrum
# - mechanism to animate arbitrary computations in rt
# - capture sreen dimensions using pure python
# - create a callback mechnism which interfaces like a rt algo block and
#   allows for metric animations used for algo development

# debug
from imp import reload
# required libs
import numpy as np
# mpl
# import matplotlib
# matplotlib.use('Qt4Agg')
from matplotlib.figure import Figure
from matplotlib.artist import getp
from matplotlib.backends.backend_qt4 import FigureManager, FigureCanvasQT, \
     new_figure_manager_given_figure

from matplotlib.lines import Line2D
# from matplotlib.backends.backend_qt4 import new_figure_manager_given_figure
# from matplotlib.backends import pylab_setup
# _backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()

from matplotlib import animation
import matplotlib.pyplot as plt
# from matplotlib import pylab

# from scipy.io import wavfile
import subprocess
import os
import gc

# visig libs
from utils import Lict, print_table, wav2np


class SigMng(object):
    '''
    Wraps an ordered map of 'signals' for easy data/vector mgmt, playback and visualization
    The mpl figure is managed as a singleton and subplots are used by default whenever
    viewing multiple signals. Numpy arrays are loaded as 'visarrays' which maintain
    references to mpl axes and artists internally and use the array data directly.
    This allows for efficient animation support with a compact interface.

    The main intent of this class is to avoid needless figure clutter, memory consumption,
    and boilerplate lines found in sig proc scripts.

    You can use this class interactively (IPython) as well as programatically
    Examples:
        >>> s = SigMng()
        >>> s.find_data('path/to/audio/files')
        ...
        found 4 'wav' files
        >>> s.show_corpus
        ...
    '''
    def __init__(self, *args):
        self._signals = Lict() # what's underneath...
        # unpack any file names that might have been passed initially
        if args:
            for i in args:
                self._sig[i] = None

        # mpl stateful objs
        # self._lines    = []
        self._fig        = None
        self._mng        = None
        self._axes_cache = Lict()
        self._arts       = []

        # to be updated by external thread
        self._cur_sample = 0
        self._cur_sig   = None

        # animation state
        # self._provision_anim = lambda : None
        # self._anim_func      = None
        self._frames_gen       = None
        # self._anim_fargs     = None
        # self._anim_sig_set   = set()
        # self._time_elapsed   = 0
        self._realtime_artists = []
        self._cursor           = None

        # animation settings
        # determines the resolution for animated features (i.e. window size)
        self._fps = 15

        # FIXME: make scr_dim impl more pythonic!
        self.w, self.h = scr_dim()
        # get the garbarge truck rolling...
        gc.enable()

    # FIXME: is this superfluous?
    # delegate as much as possible to the oid
    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError:
            return getattr(self._signals, attr)

    def __setitem__(self, *args):
        return self._signals.__setitem__(*args)

    def __getitem__(self, key):
        '''lazy loading of signals and delegation to the oid'''
        # path exists in our set
        if issubclass(self._signals[key], np.ndarray): pass
        # path exists but signal not yet loaded
        else: self._load_sig(key)
        return self._signals[key]

    def __del__(self):
        self.kill_mpl()
        self.free_cache()

    # TODO: move to Signal class
    def _load_sig(self, path):
        if isinstance(path, int):
            path = self._signals._get_key(path)
        if path in self._signals:
            # TODO: loading should be completed by 'Signal' class (i.e. format specific)
            try:
                print("loading wave file : ", os.path.basename(path))
                # read audio data and params
                sig, self.Fs, self.bd = wav2np(path)
                # (self.Fs, sig) = wavfile.read(self.flist[index])

                amax = 2**(self.bd - 1) - 1
                sig = sig/amax
                self._signals[path] = sig
                print("INFO |->",len(sig),"samples =",len(sig)/self.Fs,"seconds @ ",self.Fs," Hz")
                return path
            except:
                raise Exception("Failed to load wave file!\nEnsure that the wave file exists and is in LPCM format")
        else: raise KeyError("no entry in the signal set for key '"+str(path)+"'")

    def _prettify(self):
        '''pretty the figure in sensible ways'''
        # tighten up the margins
        self._fig.tight_layout(pad=1.03)

    def _sig2axis(self, sig_key=None):
        '''return rendered axis corresponding to a signal'''
        #FIXME: in Signal class we should assign the axes
        # on which a signal is drawn to avoid this hack?
        if not self._fig: return None
        # if not sig_key: sig_key = self._cur_sig
        sig = self[sig_key]
        try:
            for ax in self._axes_cache.values():
                for line in ax.get_lines():
                    if sig in line.get_ydata():# continue
                        return ax
                    else:
                        return None
        except ValueError:      # no easy way to compare vectors?
            return None
            # return ax
        # else:
        #     return self._sig2axis(self._cur_sig)

    getp = lambda key : getp(self._axes_cache[key])

    def _init_anim(self):
        '''
        in general we provision as follows:
        1) check if the 'current signal(s)' (_cur_signal) is shown on a figure axis
        2) if not (re-)plot them
        3) return the baseline artists which won't change after plotting (usually the time series)
        '''
        # axes = [self._sig2axis(key) for key in self._anim_sig_set]
        # ax = self._sig2axis(self._cur_sig)
        #4) do addional steps using self._provision_anim
        # self._provision_anim()
        # return the artists which won't change during animation (blitting)
        y = tuple(axes for axes in self._fig.get_axes())
        print(y[0])
        return y
        # return ax.get_lines()
        # line = vline(axis, time, colour=green)

    def _do_fanim(self, fig=None):
        '''run the function based animation once'''
        if not fig:
            fig = self._fig
        anim = animation.FuncAnimation(fig,
                                       _set_cursor,
                                       frames=self.fr,  #self._audio_time_gen,
                                       init_func=self._init_anim,
                                       interval=1000/self._fps,
                                       fargs=self._arts,
                                       blit=True,
                                       repeat=False)

        return anim
            # self._animations.appendleft(anim)
        # else: raise RuntimeError("no animation function has been set!")

    def sound(self, key, **kwargs):
        '''JUST play sound'''
        sig = self[key]
        sound4python(sig, 8e8, **kwargs)

    def play(self, key):
        '''play sound + do mpl animation with a playback cursor'''
        # sig = self[key]
        self._cur_sig = key
        ax = self._sig2axis(key)
        if not ax:
            ax = self.plot(key)

        # self._arts.append(anim_action(cursor(ax, 0), action=Line2D.set_xdata))
        # self._arts.append(
        self._cursor = cursor(ax, 10)
        # set animator routine
        # self._anim_func = self._set_cursor
        # set the frame iterator
        self._frames_gen = get_ift_gen()

        # t = threading.Thread( target=buffer_proc_stream, kwargs={'proc': p, 'callback': callback} ) #, q))
        # t.daemon = True  # thread dies with program
        # t.start()
        self._do_fanim()

    def _audio_time_gen(self):
        '''generate the audio sample-time for cursor placement
        on each animation frame'''
        # frame_step = self.Fs / self._fps    # samples/frame
        time_step = 1/self._fps
        self._audio_time = 0  # this can be locked out
        # FIXME: get rid of this hack job!
        total_time = len(self[self._cur_sig]/self.Fs)
        while self._audio_time <= total_time:
            yield self._audio_time
            self._audio_time += time_step

    def _show_corpus(self):
        '''pretty print the internal path list'''
        # TODO: show the vectors in the last column
        try:
            print_table(map(os.path.basename, self._signals.keys()))
        except:
            raise ValueError("no signal entries exist yet!?...add some first")

    # convenience attrs
    figure = property(lambda self: self._fig)
    mng = property(lambda self: self._mng)
    flist = property(lambda self: [f for f in self.keys()])
    show_corpus = property(lambda self: self._show_corpus())

    def get(self, key):
        self.__getitem__(key)

    def kill_mpl(self):
        # plt.close('all')
        self.mng.destroy()
        self._fig = None

    def close(self):
        if self._fig:
            # plt.close(self._fig)
            # self._fig.close()
            self._fig = None #FIXME: is this necessary?

    def clear(self):
        #FIXME: make this actually release memory instead of just being a bitch!
        self._signals.clear()
        # gc.collect()

    def fullscreen(self):
        '''convenience func to fullscreen if using a mpl gui fe'''
        if self._mng:
            self._mng.full_screen_toggle()
        else:
            print("no figure handle exists?")

    def add_path(self, p):
        '''
        Add a data file path to the SigMng
        Can take a single path string or a sequence as input
        '''
        if os.path.exists(p):
            # filename, extension = os.path.splitext(p)
            if p not in self:#._signals.keys():
                self[p] = None
            else:
                print(os.path.basename(p), "is already in our path db -> see grapher.SigPack.show()")
        else:
            raise ValueError("path string not valid?!")

    def plot(self, *args, **kwargs):
        '''
        can take inputs of ints, ranges or paths
        meant to be used as an interactive interface...
        returns a either a list of axes or a single axis
        '''
        axes = [axis for axis, lines in self.itr_plot(args, **kwargs)]
        self._prettify()
        if len(axes) < 2:
            axes = axes[0]
        # self.figure.show() #-> only works when using pyplot
        return axes

    def itr_plot(self, items, **kwargs):
        '''
        a lazy plotter to save aux space?...doubtful
        should be used as the programatic interface to _plot
        '''
        paths = []
        for i in items:
            # path string, add it if we don't have it
            if isinstance(i, str) and i not in self:
                self.add_path(i)
                paths.append(i)
            elif isinstance(i, int):
                paths.append(self._get_key(i))  # delegate to oid (is this stupid?)

        # plot the paths (composed generator)
        # return (axis,lines for axis,lines in self._plot(paths, **kwargs))
        for axis,lines in self._plot(paths, **kwargs):
            yield axis, lines

    def _plot(self, keys_itr, start_time=0, time_on_x=True, singlefig=True, title=None):
        '''
        plot generator - uses 'makes sense' figure / axes settings
        inputs: keys_itr -> must be an iterator over names in self.keys()
        '''
        # FIXME: there is still a massive memory issue when making multiple plot
        # calls and I can't seem to manage it using the oo-interface or
        # pyplot (at least not without closing the figure all the time...lame)

        if isinstance(keys_itr, list):
            keys = keys_itr
        else:
            keys = [i for i in keys_itr]

        # create a new figure and format
        if not singlefig or not (self._fig and self._mng.window):

            # using mpl/backends.py pylab setup (NOT pylab)
            # self._mng = new_figure_manager(1)
            # self._mng.set_window_title('visig')
            # self._fig = self._mng.canvas.figure

            # using pylab
            # pylab and pyplot seem to be causing mem headaches?
            # self._fig = pylab.figure()
            # self._mng = pylab.get_current_fig_manager()

            # using pyplot
            self._fig = plt.figure()
            self._mng = plt.get_current_fig_manager()

            # using oo-api directly
            # self._fig = Figure()
            # self._canvas = FigureCanvasQT(self._fig)
            # self._mng = new_figure_manager_given_figure(self._fig, 1)
            # self._mng = FigureManager(self._canvas, 1)
            # self._mng = new_figure_manager_given_figure(1, self._fig)

            self._mng.set_window_title('visig')
        else:
            # for axis in self.figure.get_axes():
            #     axis.clear()
            #     gc.collect()
                # for line in axis.get_lines():
                    # line.clear()
                    # gc.collect()
            self.figure.clear()
            gc.collect()

        # draw fig
        # TODO: eventually detect if a figure is currently shown?
        # draw_if_interactive()

        # set window to half screen size if only one signal
        if len(keys) < 2: h = self.h/2
        else: h = self.h
        # try:
        self._mng.resize(self.w, h)
        # except: raise Exception("unable to resize window!?")
        # self._fig.set_size_inches(10,2)
        # self._fig.tight_layout()

        # title settings
        font_style = {'size' : 'small'}

        self._axes_cache.clear()
        # main plot loop
        for icount, key in enumerate(keys):
            # always set 'curr_sig' to last plotted
            self._cur_sig = key
            sig = self[key]
            slen  = len(sig)

            # set up a time vector and plot
            t     = np.linspace(start_time, slen / self.Fs, num=slen)
            ax    = self._fig.add_subplot(len(keys), 1, icount + 1)

            # maintain the key map to our figure's axes
            self._axes_cache[key] = ax

            lines = ax.plot(t, sig, figure=self._fig)
            ax.set_xlabel('Time (s)', fontdict=font_style)

            if title == None: title = os.path.basename(key)
            ax.set_title(title, fontdict=font_style)

            # ax.figure.canvas.draw()
            ax.figure.canvas.draw()
            yield (ax, lines)

    def find_wavs(self, sdir):
        '''find all wav files in a dir'''
        for i, path in enumerate(file_scan('.*\.wav$', sdir)):
            self[path] = None
            print("found file : ", path)
        print("found", len(self.flist), "files")

    # example callback
    def print_cb(self, *fargs):
        for i in fargs:
            print(i)

    def register_callback(self):
        pass


class visarray(np.ndarray):
    '''
    Extend the standard ndarray to include metadata about
    the contained vector such as the sample rate 'Fs',
    bitdepth, etc. and maintain an mpl artist which refers
    to the array directly.

    For plotting a user should utilize the mpl OO interface
    by adding the artist to an extant mpl axis via axis.add_artist()

    Motivations for this subclass include:
        1) aux memory is saved such that data is not duplicated inside mpl
           (i.e. ax.plot(t, array) creates a new array/Artist each call
           see matplolib.artist.plot)
        2) changes to the array data are rendered on the next canvas.draw()
           (useful for rt animations)
        3) the instance maintains the same interface as a normal ndarray

    Addtionally, this class offers facilities for loading arbitray (cough audio)
    data files into np arrays. A last resort brute force procedure is
    first convert to wav using a system util (ex. sox) and then load from
    wav to numpy array from the wave module.

    'new_from_template' views onto this array are instantiated with
    a new internal artist who reference the view data.

    For more info on subclassing ndarrys see here:
    <link here>
    '''
    _artist = Line2D

    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None,
                # extra visarray args
                Fs=None, artist_type=Line2D):

        # call the parent constructor
        obj = np.ndarray.__new__(cls, shape, dtype,
                                buffer, offset, strides, order)

        # record the sample rate
        obj.Fs = Fs
        # instance our internal artist
        obj._artist = artist_type(self, self.size)
        return obj
    # artist = property(lambda self: self._fill_artist(np.arange(self), self))

    def _get_artist(self):
        return self._artist

    def _set_artist(self, artist):
        self._artist = artist

    artist = property(_get_artist, _set_artist)

    def __array_finalize__(self, obj):
        # for each case self,obj are:
        # s,'none' when called from ndarray.__new__ i.e. s = visarray()
        # s,nparr when called by s = nparr.view(visarray)
        # s,v when assigned by s = v[:4] and type(v) == visarray

        # called from __new__
        if obj is None: return

        # add attributes on new arrays (for all cases minus instantiation)
        self.Fs = getattr(obj, 'Fs', None)

    def _load_data(self, path):
        try:
            print("loading wave file : ",os.path.basename(path))
            # read audio data and params
            sig, self.Fs, self.bd = wav2np(path)
            # (self.Fs, sig) = wavfile.read(self.flist[index])

            # max for signed integer data
            amax = 2**(self.bd - 1) - 1
            sig = sig/amax
            self._signals[path] = sig
            print("INFO |->",len(sig),"samples =",len(sig)/self.Fs,"seconds @ ",self.Fs," Hz")
            return path
        except:
            raise Exception("Failed to load wave file!\nEnsure that the wave file exists and is "
                            "in LPCM format")

    # def __getattr__(self, attr):
    #     return getattr(self._array, attr)

    def _detect_fmt(selft):
        raise NotImplementedError('Needs to be implemented by subclasses to'
                                  ' actually detect file format.')


def cursor(axis, time, colour='r'):
    '''add a vertical line @ time (...looks like a cursor)
    here the x axis should be a time vector such as created in SigMng._plot
    the returned line 'l' can be set with l.set_data([xmin, xmax], [ymin, ymax])'''
    return axis.axvline(x=time, color=colour)


def file_scan(re_literal, search_dir, method='find'):
    if method == 'find':
        try:
            found = subprocess.check_output(["find", search_dir, "-regex", re_literal])
            paths = found.splitlines()

            # if the returned values are 'bytes' then convert to strings
            str_paths = [os.path.abspath(b.decode()) for b in paths]
            return str_paths

        except subprocess.CalledProcessError as e:
            print("scanning logs using 'find' failed with output: " + e.output)

    elif method == 'walk':
        # TODO: os.walk method
        print("this should normally do an os.walk")
    else:
        print("no other logs scanning method currently exists!")


def scr_dim():
    # TODO: find a more elegant way of doing this...say using
    # figure.get_size_inches()
    # figure.get_dpi()
    dn = os.path.dirname(os.path.realpath(__file__))
    bres = subprocess.check_output([dn + "/screen_size.sh"])
    dims = bres.decode().strip('\n').split('x')  # left associative
    return tuple([int(i.strip()) for i in dims if i != ''])


# TODO: should try out pyunit here..
def get_ss():
    # example how to use the lazy plotter
    ss = SigMng()
    ss.find_wavs('/home/tyler/clips/')
    # wp.plot(0)
    return ss


def anim_action(artist, data=None, action=Line2D.set_data):
    if data:
        return lambda iframe, data: artist.action(iframe, data)


class Animatee(object):
    '''
    base class container which provides an interface for
    artists which animate under a sequened timing (i.e. TimedAnimations)
    '''
    def __init__(self, axis, artist=Line2D, data=None):
        self.axis = axis
        self.artist = artist
        self.data = data
        return self.anim_init()

    def anim_init(self):
        '''
        define the initialization for this animatee
        '''
        pass

    def anim_step(self, iframe, itime):
        '''
        define the action to be performed on our artist
        each animation step
        '''
        self.artist.set_data(iframe, 1)


class Cursor(Animatee):
    '''a cursor for playback'''
    def anim_init(self):
        '''create our vertical line artist'''
        self.artist = self.artist.axis.axvline(0, color='r')

    def anim_step(self, iframe, itime):
        self.artist.set_xdata(itime)


# def _set_cursor(iframe, *rt_anims):
def animate(ift, *animatees):
    '''perform animation step using the frame sequence tuple
    and the provided iter of animatees
    '''
    # print(str(time.time()))
    # for array,line in fargs:
    #   line.set_ydata(array)
    # TODO: can we do a map using the Animatee 'animate' function?
    # TODO: animatees needs to be a generator which yeilds systems in order
    for animatee in animatees:
        animatee.anim_step(*ift)#, animatee.data)
    return [animatee.artist for animatee in animatees]
        # cursor_line = f
        # cursor_line.set_xdata(float(iframe))
    # self._cursor.set_data(float(sample_time), [0,1])
    # self.figure.show()
    # return self._cursor


# script interface
if __name__ == '__main__':
    print("play with instance 'ss'\ne.g. ss.show_corpus...")
    ss = get_ss()
