'''
animation machinery
'''
import time
from matplotlib import animation, pyplot as plt
from utils import wav2np


def get_ift_gen(length, Fs, fps=15, init=0):
    '''
    create a generator which delivers tuples
    with (frameindex, timevalue) each frame
    of the animation
    Inputs:
        length - length of the input input animation sequence
        Fs     - sample rate of the input sequence data

    Optionals:
        fps    - frame rate (default 15 fps)
        init   - initial frame sequence number (default 0)
    '''
    sample_step = Fs / fps      # samples/frame
    time_step = 1/fps           # seconds
    total_time = length / Fs
    # init_val = 0                # this can be locked out?
    # _audio_time = 0             # this can be locked out?
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


def animate(array, fig, func, fs, cb_args=None, fps=15, frames=None,
            init=None):
    if not frames:
        frames = get_ift_gen(len(array), fs, fps)

    anim = animation.FuncAnimation(
        fig,
        func,
        frames=frames,
        init_func=init,
        interval=1000/fps,
        fargs=cb_args,
        blit=True,
        repeat=False
    )
    return anim


if __name__ == '__main__':
    import sys
    fname = sys.argv[1]  # first arg is the wave file name
    array, fs, bd = wav2np(fname)
    artists = plt.plot(array)
    line = artists[0]
    axes = line.axes
    fig = line.get_figure()
    cursor = axes.axvline(0, color='r')

    def callback(framedata):
        isample, itime = framedata
        cursor.set_xdata(isample)
        return (cursor,)  # must return collection of artists

    init = lambda: artists
    anim = animate(array, fig, callback, fs, init=init)
