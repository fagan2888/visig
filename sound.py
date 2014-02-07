'''
sound4python module

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
