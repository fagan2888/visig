'''
numpy extensions
'''
import numpy


class visarray(numpy.ndarray):
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
