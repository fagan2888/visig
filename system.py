'''
an algorithm can be applied against at most one input signal
a signal can have multiple algorithms producing computations on it

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


def rms(x) -> 'x.ndtype':
    return np.sqrt(x.dot(x)/x.size)


def rms2(x) -> float:
    # fast but more accurate?
    return la.norm(x) / np.sqrt(x.size)


def rms3(x) -> float:
    return np.sqrt(np.vdot(x, x)/x.size)


def db(x):
    20*np.log10(x)


# @params(Tw=1.0)
@visig.plot(loc=(2, 1))
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


@visig.plot  #(fig=1, row=0, col=1)
def moving_avg(a, n=3):
    '''Compute a moving average of window length `n`
    '''
    cs = np.cumsum(a, dtype=float)
    # do s[k] = cs[k+n] - cs[k], where k >= n
    cs[n:] = cs[n:] - cs[:-n]
    return cs[n - 1:] / n


@visig.plot
def agc(src, tg: -100. < float < 0. = -20.):
    '''Automatic gain control

    Normalize the input to a target gain using a long
    term iir psuedo-windowed average.
    '''
    return tg / rmean(np.abs(src)) * src


if __name__ == '__main__':
    # @visig.source
    # def sig():
    import sys
    from utils import wav2np
    wavfile = sys.argv[1]
    sig, fs, bd = wav2np(wavfile)
    return sig, fs

    # from visig import frameify
    @visig.sink(sig=sig, Fs=fs)
    def agc(sig):
        # offline processing
        # from visig.np import abs
        absolute = np.abs(sig)
        mavg = moving_avg(absolute, n=100)
        ravg = rmean(absolute, fs)
        agc_out = agc(sig)
        return agc_out, ravg

    # visig.run(frameify=True)
