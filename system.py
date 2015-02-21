'''
an algorithm can be applied against at most one input signal
a signal can have multiple algorithms producing computations on it

make a request to this manager for an algorithm and apply it to an input

This class organizes a collection of signal processing systems into a directed
graph to produce an algorithm. An algorithm can reports feature computations
on the input signal and may produce an ouput signal

A class which implements a signal processing system T such that
in y[n] = T{x[n]}.
Use this class as a module which may be serialized/parallelized
with other modules in order to build a signal processing chain.
'''
import numpy as np
from numpy import linalg as la


def ulaw(src, u=255):
    'u-law encoder'
    return np.sign(src) * np.log(1 + u * np.abs(src)) / np.log(1 + u)


# @sparams(Tw=1.0)
def rmean(src, Fs, Tw=1.0):
    '''Rolling mean of window length `Tw * frame_size`
    '''
    M = np.ceil(Tw * Fs)
    a = 1/M
    b = 1-a
    ax = np.zeros(src.size)
    for n, x in enumerate(src):
        ax[n] = a*x + b*(ax[n-1])
    return ax


def moving_avg(a, n=3):
    '''Compute a moving average of window length `n`
    '''
    cs = np.cumsum(a, dtype=float)
    # do s[k] = cs[k+n] - cs[k], where k >= n
    cs[n:] = cs[n:] - cs[:-n]
    return cs[n - 1:] / n


def rms(x) -> float:
    return np.sqrt(x.dot(x)/x.size)


def rms2(x) -> float:
    # fast but more accurate?
    return la.norm(x) / np.sqrt(x.size)


def rms3(x) -> float:
    return np.sqrt(np.vdot(x, x)/x.size)


def db(x):
    20*np.log10(x)


# def agc(
#     src,
#     absmean: 'src -> abs',
#     tg: -100. < float < 0. = -20.,
# ):
#     '''Automatic gain control
#     '''
#     gains = tg / rmean
#     return gains * src  # normalize to target gains



# can we build a mechanism which composes/cascades systems?
# class Agc(System):
#     G = -20.0  # dBFS

#     def process(self, in_frame, out_frame, *args):
#         # mx = self.get_feature('QuickRollingMean')
#         ax = mx(abs(in_frame), Tw*Fs)
#         gains = tg/ax
#         return (frame * gains, gains, ax)
#         return agc(array)

if __name__ == '__main__':
    _src = np.linspace(0, 1, num=512)
    import sys
    from utils import wav2np
    wavfile = sys.argv[1]
    sig, fs, bd = wav2np(wavfile)

    iframes = np.append(np.arange(0, sig.size, 512), sig.size)
    for start, stop in zip(iframes, iframes[1:]):
        frame = sig[start:stop]
    abs = np.abs(sig)
    mavg = moving_avg(abs, n=100)
    ravg = rmean(abs, fs)
