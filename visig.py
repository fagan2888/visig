#!/usr/bin/env python3
'''
Visig   - visualize and animate signals using mpl

author  : Tyler Goodlet
email   : tgoodlet@gmail.com
website : http://tgoodlet.github.com
license : BSD
Please feel free to use and modify this, but keep the above information.
'''
# TODO;
# - consider using sox conversion in this module so we can open
#   arbitrarly formatted audio files as numpy arrays
# - 'Signal' class which is delegated all data unpacking and metadata maintenance
# - animation for simple playback using a cursor on the chosen axes (add mixing?)
# - animation for rt spectrum
# - mechanism to animate arbitrary metric computations in rt
# - capture sreen dimensions using pure python
# - create a callback mechnism which interfaces like a rt algo block and
#   allows for metric animations used for algo development

# required libs
from imp import reload
import numpy as np

# mpl
# import matplotlib
# matplotlib.use('Qt4Agg')
from matplotlib.figure import Figure
from matplotlib.artist import getp
from matplotlib.backends.backend_qt4 import FigureManager, FigureCanvasQT, \
     new_figure_manager_given_figure
# from matplotlib.backends.backend_qt4 import new_figure_manager_given_figure
# from matplotlib.backends import pylab_setup
# _backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()

from matplotlib import animation
import matplotlib.pyplot as plt
# from matplotlib import pylab

#from scipy.io import wavfile
import subprocess, os, gc

# visig libs
from utils import Lict, print_table

class SigMng(object):
    '''
    Wraps an ordered map of signals for easy data/vector mgmt, playback and visualization
    The mpl figure is managed as a singleton and subplots are used whenever viewing multiple signals
    The main intent is to avoid needless figure clutter and boilerplate lines found in sig proc scripts

    Use this class interactively (IPython) as well as programatically
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
        self._cur_sig    = None
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
        self._fps = 15   # determines the resolution for animated features (i.e. window size)

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
        if isinstance(self._signals[key], np.ndarray): pass
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
            #TODO: loading should be completed by 'Signal' class (i.e. format specific)
            try:
                print("loading wave file : ",os.path.basename(path))
                # read audio data and params
                sig, self.Fs, self.bd = wav_2_np(path)
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

    def _do_fanim(self):
        '''run the function based animation once'''
        # if self._anim_func:
        anim = animation.FuncAnimation(self._fig,
                                       _set_cursor,
                                       frames=itime,#self._audio_time_gen,
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

        self._arts.append(anim_action(cursor(ax, 0), action=Line2D.set_xdata))
        self._cursor = cursor(ax, 10)
        # set animator routine
        # self._anim_func = self._set_cursor
        # set the frame iterator
        self._frames_gen = itime()

        # t = threading.Thread( target=buffer_proc_stream, kwargs={'proc': p, 'callback': callback} ) #, q))
        # t.daemon = True  # thread dies with program
        # t.start()
        self._do_fanim()


    def _audio_time_gen(self):
        '''generate the audio sample-time for cursor placement
        on each animation frame'''
        # frame_step = self.Fs / self._fps    # samples/frame
        time_step = 1/self._fps
        self._audio_time = 0                # this can be locked out
                                        # FIXME: get rid of this hack job!
        total_time = len(self[self._cur_sig]/self.Fs)
        while self._audio_time <= total_time:
            yield self._audio_time
            self._audio_time += time_step

    def _show_corpus(self):
        '''pretty print the internal path list'''
        # TODO: show the vectors in the last column
        try: print_table(map(os.path.basename, self._signals.keys()))
        except: raise ValueError("no signal entries exist yet!?...add some first")

    # convenience attrs
    figure      = property(lambda self: self._fig)
    mng         = property(lambda self: self._mng)
    flist       = property(lambda self: [f for f in self.keys()])
    show_corpus = property(lambda self : self._show_corpus())
    def get(self, key): self.__getitem__(key)

    def kill_mpl(self):
        # plt.close('all')
        self.mng.destroy()
        self._fig = None

    def close(self):
        if self._fig:
            # plt.close(self._fig)
            self._fig.close()
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
        axes = [axis for axis,lines in self.itr_plot(args, **kwargs)]
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
                paths.append(self._get_key(i)) # delegate to oid (is this stupid?)

        # plot the paths (composed generator)
        # return (axis,lines for axis,lines in self._plot(paths, **kwargs))
        for axis,lines in self._plot(paths, **kwargs):
            yield axis, lines

    def _plot(self, keys_itr, start_time=0, time_on_x=True, singlefig=True, title=None):
        '''
        plot generator - uses 'makes sense' figure / axes settings
        inputs: keys_itr -> must be an iterator over path names in self.keys()
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
        # self.flist = file_scan('.*\.wav$', sdir)
        for i, path in enumerate(file_scan('.*\.wav$', sdir)):
            self[path] = None
            print("found file : ",path)
        print("found", len(self.flist), "files")

    # example callback
    def print_cb(self, *fargs):
        for i in fargs:
            print(i)

    def register_callback(self):
        pass


class Signal(np.ndarray):
    '''
    extend the standard ndarray to include metadata about the contained vector
    such as the sample rate 'Fs' and maintain an artist which refers to the data directly
    instead of making a copy as with mpl's ax.plot(t, array)...

    wrapper class for loading arbitray data files into np arrays
    brute force procedure is first convert to wav using a system util (ex. sox)
    and then load from wav to numpy array
    '''
    def __init__(self, array=None, data_file=None, artist_cls=Line2D):
        self._Fs = None
        self.artist = artist_cls#(np.arange(len(self), ydata=self)
        # self._detect_fmt(file_path)

    artist = property(lambda self: self._fill_artist(np.arange(self), self))

    def _load_sig(self, path):
        try:
            print("loading wave file : ",os.path.basename(path))
            # read audio data and params
            sig, self.Fs, self.bd = wav_2_np(path)
            # (self.Fs, sig) = wavfile.read(self.flist[index])

            amax = 2**(self.bd - 1) - 1
            sig = sig/amax
            self._signals[path] = sig
            print("INFO |->",len(sig),"samples =",len(sig)/self.Fs,"seconds @ ",self.Fs," Hz")
            return path
        except:
            raise Exception("Failed to load wave file!\nEnsure that the wave file exists and is in LPCM format")

    def __getattr__(self, attr):
        return getattr(self._array, attr)

    def _detect_fmt(selft):
        raise NotImplementedError('Needs to be implemented by subclasses to'
                                  ' actually detect file format.')
import wave
def wav_2_np(f):
    ''' use the wave module to make a np array'''
    wf = wave.open(f, 'r')
    Fs = wf.getframerate()

    # bit depth calc
    bd = wf.getsampwidth() * 8
    frames = wf.readframes(wf.getnframes())

    # hack to read data using array protocol type string
    dt = np.dtype('i' + str(wf.getsampwidth()))
    sig = np.fromstring(frames, dtype=dt)
    wf.close()
    return (sig, Fs, bd)

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
        #TODO: os.walk method
        print("this should normally do an os.walk")
    else:
        print("no other logs scanning method currently exists!")

def scr_dim():
    #TODO: find a more elegant way of doing this...say using
    # figure.get_size_inches()
    # figure.get_dpi()
    dn = os.path.dirname(os.path.realpath(__file__))
    bres = subprocess.check_output([dn + "/screen_size.sh"])
    dims = bres.decode().strip('\n').split('x')  # left associative
    return tuple([int(i.strip()) for i in dims if i != ''])

# from the sound4python module
'''
Copyright (C) 2013 dave.crist@gmail.com
edited and extended by tgoodlet@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

try: import tempfile, wave, signal, struct, threading, sys#, contextlib
except ImportError as imperr : print(imperr)

# but Popen uses the alias DEVNULL anyway...? (legacy...damn)
FNULL = open(os.devnull,'w')

#TODO: consider checking out the nipype 'run_command' routine for more ideas...?
# class ProcessLauncher(object):
#     '''base class for process launchers'''

#     def __init__(self, **args):
#         '''passing a callback implies you want to pipe output from the process
#         and the callback '''
#         self._settings = args

def launch_without_console(args, pipe_output=False):
    '''Launches args windowless and waits until finished
    Reimplement this if don't want to use the Popen class
    parameters : 'args' list of cmdline tokens, pipe_output toggle
    outputs : process instance
    '''
    # def launch_without_console(args, get_output=False):
        # """Launches args windowless and waits until finished"""
    startupinfo = None

    if 'STARTUPINFO' in dir(subprocess):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    # if pipe_output:
    if self.callback:
        # create process
        return subprocess.Popen(args,
                         # bufsize=1,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         startupinfo=startupinfo)
        # return p
    else:
        return subprocess.Popen(args,
                                stdin=subprocess.PIPE,
                                stdout=FNULL,
                                # stderr=FNULL,
                                startupinfo=startupinfo)

    # def run(self, get_output=False):
    #     return self._launch(arg_list, get_output)

# TODO: make the subclasses and use class attributes to customize for
# each util?
class SndFileApp(object):
    '''
    simple class for wrapping sound utils
    inputs :
               app           - the arg0 part
               playback_args - a tokenized in a list of command options which allow for playback
               fmt_analyser  - also an arg list which can be used to analyse audio files

    You can additionally pass a callable 'parser' which can be used to parse the output
    from the application's std streams
    '''
    _cmd = 'sox'
    _playback_args = []
    def __init__(self, app, playback_args, fmt_analyser=None,
                launcher=launch_without_console,
                parser=None):

        self._cmdline = []
        self._cmdline.append(app)
        self._cmdline.extend(playback_args)
        self._proc_launcher = launcher
        if fmt_analyser:
            self._fmt_anal = fmt_analyser
        if parser:
            self._parser = parser

    def read(self, f):
        '''read in a sound file using the sound app utilities'''
        pass

    def launch(self, callback=None, parser=lambda arg : arg):
        '''launch the sound app as a subprocess
        provide command output parsing if a callback is provided'''
        get_output = lambda : callback is not None
        prs = self._parser or parser
        p = self._proc_launcher(self._cmdline, get_output())
        if callback:
            # create a thread to handle app std stream parsing
            t = threading.Thread(target=buffer_proc_stream,
                                 kwargs={'proc': p, 'callback': callback, 'parser' : prs})
            t.daemon = True  # thread dies with program
            t.start()
            self._output_handler = t
        return p

    def analyze(self, f):
        '''sublcass method to analyze a file'''
        raise NotImplementedError("must implement the file analysis in the child class!")

    def sound(self, itr, Fs, **args):
        '''use sound4python module for playback'''
        sound4python(itr, Fs, app=self, **args)

# app parser funcs
def parse_sox_cur_time(s):
    '''
    s : string or buffer?
    parser funcs must take in a string or buffer and provide a string output
    '''
    val = s.strip('\n')
    return val
#TODO
def parse_aplay(s):
    pass
def parse_sndfile(s):
    pass

# audio (linux) app cmds
# TODO:
# - check for app existence in order
# - extend the SndApp class to contain method calls for generic analysis
#   and multichannel playback (sox supports this)
# - should this class contain a parser plugin mechanism?

snd_utils            = Lict()
snd_utils['sox']     = SndFileApp('sox', playback_args=['-','-d'], parser=parse_sox_cur_time)
snd_utils['aplay']   = SndFileApp('aplay', playback_args=['-vv', '--'], parser=parse_aplay)
snd_utils['sndfile'] = SndFileApp('sndfile-play', playback_args=['-'], parser=parse_sndfile)

def get_snd_app():
    '''get the first available sound app'''
    for k,v in snd_utils.items():
        if k in os.defpath:
            return v
        else:
            continue
    # for app in snd_utils:
    #     arg0 = app.

def sound4python(itr, Fs, bitdepth=16, start=0, stop=None,
          # app_name='sox',
          app=None,
          autoscale=True,
          level =-18.0,
          callback=None):
    '''
    a python sound player which delegates to a system (Linux) audio player

    params:
            itr          : input python iterable for playback
            Fs           : sample rate of data
            start/stop   : start/stop vector indices
            app_name     : system app of type SndApp used for playback
                           current available options are sox, alsa play, and lib-sndfile play
            autoscale    : indicates to enable playback at the provided 'level'
            level        : volume in dBFS (using an rms calc?)
            callback     : will be called by the snd app with (parser) output passing in 'fargs'
    '''
# TODO: move these imports out of here?
    try:
        import numpy as np
        foundNumpy = True
    except ImportError as imperr:
        foundNumpy = False;
        print(imperr)

    print("playing from", start,"until", stop)
    # set start sample
    start = start * Fs
    # set stop sample
    if not stop:
        stop = len(itr)
    else:
        stop = stop * Fs

    # slicing should work for most itr
    itr = itr[start:stop]

    #for now, assume 1-D iterable
    # mult = 1
    if autoscale:
        # multiplier to scale signal to max volume at preferred bit depth
        mxval = 2**(bitdepth - 1)           # signed 2's comp
        A = 10**(level/20.)                 # convert from dB
        mult = A * float(mxval) / max(itr)

    #create file in memory
    memFile = tempfile.SpooledTemporaryFile()

    # create wave write objection pointing to memFile
    waveWrite = wave.open(memFile,'wb')
    waveWrite.setsampwidth(int(bitdepth/8))  # int16 default
    waveWrite.setnchannels(1)           # mono  default
    waveWrite.setframerate(Fs)
    wroteFrames = False

    # utilize appropriate data type
    dt = np.dtype('i' + str(bitdepth))
    # try to create sound from NumPy vector
    if foundNumpy:
        if type(itr) == np.ndarray:
            if itr.ndim == 1 or itr.shape.count(1) == itr.ndim - 1:
                waveWrite.writeframes( (mult*itr.flatten()).astype(dt).tostring() )
                wroteFrames=True

        else: # we have np, but the iterable isn't a vector
            waveWrite.writeframes( (mult*np.array(itr)).astype(dt).tostring() )
            wroteFrames=True

    if not wroteFrames and not foundNumpy:
        # FIXME: how to set playback bitdepth dynamically using this method?
        # -> right now this is hardcoded to bd=16
        # python w/o np doesn't have "short"/"int16", "@h" is "native,aligned short"
        waveWrite.writeframes( struct.pack(len(itr)*"@h", [int(mult*itm) for  itm in itr]) )
        wroteFrames=True

    if not wroteFrames:
        print("E: Unable to create sound.  Only 1D numpy arrays and numerical lists are supported.")
        waveWrite.close()
        return None

    # configure the file object, memFile, as if it has just been opened for reading
    memFile.seek(0)

    # getting here means wroteFrames == True
    print("\nAttempting to play a mono audio stream of length "
          "{0:.2f} seconds\n({1:.3f} thousand samples at sample "
          "rate of {2:.3f} kHz)".format(1.0*len(itr)/Fs, len(itr)/1000., int(Fs)/1000.))
    try:
        # look up the cmdline listing
        # app = snd_utils[app_name]
        if not app:
            app = get_snd_app()

        # launch the process parsing std streams output if requested
        p = app.launch(callback)
        # p = launch_without_console(app.cmd_line, get_output=True)

        # if callback:
            # create a thread to handle app output stream parsing
            # TODO: make thread a class with more flexibility
            # t = threading.Thread( target=buffer_proc_stream, kwargs={'proc': p, 'callback': callback} ) #, q))
            # t.daemon = True  # thread dies with program
            # t.start()

            # state.join()

    except:
        # FIXME: make this an appropriate exception
        print("\nE: Unable to launch sox.")
        print("E: Please ensure that sox is installed and on the path.")
        print("E: Try 'sox -h' to test sox installation.")
        waveWrite.close()
        return None
    try:
        # deliver data to process (normally a blocking action)
        p.communicate(memFile.read())
        print(app,"communication completed...")
        # p.wait()

    except:
        # FIXME: make this an appropriate exception
        print("E: Unable to send in-memory wave file to stdin of sox subprocess.")
        waveWrite.close()
        return None

#os.kill(p.pid,signal.CTRL_C_EVENT)
#end def sound(itr,samprate=8000,autoscale=True)

# # threaded class to hold cursor state info
# class SoxState(thread.
# class ProcOutput(threading.Thread):
#     def __init__(self, proc):
#         self.p = proc
#         self.stdout = None
#         self.stderr = None
#         threading.Thread.__init__(self)
# def print_stdout():
#     for line in sys.stdout.readline():
#         print line

def buffer_proc_stream(proc,
                       # deque,
                       std_stream='stderr',
                       parser=lambda arg : arg,
                       callback=print):
    '''Poll process for new output until finished pass to callback
       parser is an identity map if unassigned'''

    for b in unbuffered(proc, stream=std_stream):
        # deque.appendleft(parser(b))
        callback(parser(b))

    # stream = proc.stderr
    # nbytes = 1
    # # stream = getattr(proc, stream)
    # read = lambda : os.read(stream.fileno(), nbytes)
    # while proc.poll() is None:
    #     out = []
    #     # read a byte
    #     try:
    #         last = read().decode()
    #         print(last)
    #     # in case the proc closes and we don't catch it with .poll
    #     except ValueError:
    #         print("finished piping...")
    #         break

        # try:
        #     # read a byte at a time until eol...
        #     while last not in newlines:
        #         out.append(last)
        #         last = read().decode()
        # except:
        #     print("finished piping...")
        #     break
        # else:
        #     print(out)
            # out.clear()

# Unix, Windows and old Macintosh end-of-line
newlines = ['\n', '\r\n', '\r']
def unbuffered(proc, stream='stdout', nbytes=1):
    '''
    down and dirty unbuffered byte stream generator
    which reads the explicit fd
    (since the built-ins weren't working for me)
    '''
    stream = getattr(proc, stream)
    read = lambda : os.read(stream.fileno(), nbytes)
    while proc.poll() is None:
        out = []
        # read a byte
        last = read().decode()
        try:
            # read bytes until eol...
            while last not in newlines:
                out.append(last)
                last = read().decode()
            if len(out) == 0: continue
        # in case the proc closes and we don't catch it with .poll
        except ValueError:
            print("finished piping...")
            break
        else:
            # yield ''.join(out)
            yield out
            # out.clear()

def _in_ipython():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True

def ipy():
    '''if possible, launch ipython'''
    if _in_ipython():
        pass
    else:
        print("\nattempting to start the ipython shell...\n")
        try: from IPython import embed
        except ImportError as imperr : print(imperr)
        # start IPython whilst maintaining the current namespace and scope
        return embed()

# TODO: should try out pyunit here..
def test():
    # example how to use the lazy plotter
    ss = SigMng()
    ss.find_wavs('/home/tyler/code/python/wavs/')
    # wp.plot(0)
    return ss

from matplotlib.lines import Line2D

def anim_action(artist, data=None, action=Line2D.set_data):
    if data:
        return lambda iframe, data : artist.action(iframe, data)

class Animatee(object):
    '''
    container which provides a standard interface to
    animate artists under a sequened timing
    '''
    def __init__(self,  data, artist=Line2D, action=None):
        self.data = data
        self.artist = artist

    def animate(self, iframe, data):
        '''this can be overidden'''
        self.artist.set_ydata(data)

# def _set_cursor(iframe, *rt_anims):
def animate_with_data(ift, *rt_anims):
    '''
    perform animation step using the frame sequence value and cursor line
    cursor artist must be first element in *fargs
    '''
# TODO? place a lock here?
    # print(fargs)
    # print(iframe)
    # print(str(time.time()))
    # print(fargs)
    # for array,line in fargs:
    #   line.set_ydata(array)
    # TODO: can we do a map using the Animatee 'animate' function?
    for animatee in rt_anims:
        animatee.animate(ift)#, animatee.data)
    return rt_anims
        # cursor_line = f
        # cursor_line.set_xdata(float(iframe))
    # self._cursor.set_data(float(sample_time), [0,1])
    # self.figure.show()
    # return self._cursor

import time
_fps = 15
_Fs  = 48000
def get_ift_gen(fps, length, Fs, init=0):
    sample_step = Fs / fps    # samples/frame
    time_step = 1/fps        # seconds
    total_time = length / Fs
        # init_val = 0                # this can be locked out?
        # _audio_time = 0                # this can be locked out?
                                       # FIXME: get rid of this hack job!
    now = time.time()
    itime = isample = init
    # total_time = len(ss[0])/_Fs
    while itime <= total_time:
        yield isample, itime
        itime += time_step
        isample += sample_step
    later = time.time()
    print("total time to animate = "+str(later-now))


# return axes rendering the signal
# ss = SigMng()
# ss.find_wavs('/home/tyler/code/python/wavs/')
# a = ss.plot(0)
# cursor line added to axis, line artist returned
# lcur = cursor(a, 0)
# line = ax.get_line

def test_fanim(ss):
# call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(ss.figure, _set_cursor, init_func=init, fargs=[lcur],
                                   frames=itime, interval=1000/_fps, repeat=False, blit=True)

    return anim
    # return None

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    print(i)
    return line,

# initialization function: plot the background of each frame
def init():
    l = a.get_axes()
    # line.set_data([0,1,4], [1,4,6])
    return l,

# FEATURES:
# define the plotting routine in the metric? for example use stem instead of plot?

# script interface
if __name__ == '__main__':
    # ss = test()
    print("play with instance 'ss'\ne.g. ss.show_corpus...")
    shell = ipy()
    # anim = test_fanim(ss)
