'''
handy stuff
'''
from collections import OrderedDict, deque
import numpy
import wave


def itr_iselect(itr, *indices):
    '''select values from an itr by index
    '''
    i_set = set(indices)
    return (e for (i, e) in enumerate(itr) if i in i_set)


def print_table(itr, field_header=['signal files'], delim='|'):
    '''Naively pretty print iterable in a column
    '''
    itr = [i for i in itr]
    max_width = max(len(field) for field in itr)
    widths = iter(lambda: max_width, 1)  # an infinite width generator

    # print field title/headers
    print('')
    print('index', delim, '',  end='')
    for f, w in zip(field_header, widths):
        print('{field:<{width}}'.format(field=f, width=w), delim, '', end='')
    print('\n')

    # print rows
    for row_index, row in enumerate(itr):
        # print index
        print('{0:5}'.format(str(row_index)), delim, '', end='')
        print('{r:<{width}}'.format(r=row, width=max_width), delim, '', end='')
        print()
        # # print columns
        # for col, w in zip(row, widths):
        #     print('{column:<{width}}'.format(column=col, width=w), delim, '', end='')
        # print()


class Lict(OrderedDict):
    '''An ordered, int subscriptable dict
    '''
    def __getitem__(self, key):
        # FIXME: how to handle slices more correctly?
        if isinstance(key, slice):
            return list(self.values())[key]
        elif key in self:
            return OrderedDict.__getitem__(self, key)

        # FIXME: is this the fastest way to implement this?
        # if it's an int, iterate the linked list and return the value
        elif isinstance(key, int):
            return self[self._get_key(key)]
            # return self._setget_value(key)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = self._get_key(key)
        OrderedDict.__setitem__(self, key, value)

    def _get_key(self, key):
        '''get the key for a given index
        '''
        # check for out of bounds
        l = len(self)
        if l - 1 < key or key < -l :
            raise IndexError("index out of range, len is " + str(l))

        # get the root of the doubly linked list (see OrderedDict)
        root = self._OrderedDict__root
        curr = root.next
        act = lambda link : link.next
        if key < 0 :  # handle -ve indices too
            key += 1
            curr = root.prev
            act = lambda link : link.prev
        # traverse the linked list for our element
        for i in range(abs(key)):
            curr = act(curr)
        return curr.key


# mpl stuff
def label_ymax(axis, x, label):
    '''add an optional label to the line @ ymax
    '''
    # use ylim for annotation placement
    mx = max(axis.get_ylim())
    ret = axis.annotate(label,
                  xy=(x, mx),
                  xycoords='data',
                  xytext=(3, -10),
                  textcoords='offset points')
                  # arrowprops=dict(facecolor='black', shrink=0.05),
                  # horizontalalignment='right', verticalalignment='bottom')
    return ret


def axes_max_y(axes):
    '''Get the max value from the available lines in param:`axes`
    for annotation placement
    (a very naive impl)
    '''
    lines = axes.get_lines()
    mx = 0
    for line in lines:
        lm = max(line.get_ydata())
        if lm > mx:
            mx = lm
    return mx


def wav2np(fname):
    '''Load wave file at param:`fname` into a numpy array.
    Returns a tuple (ndarray, samplerate, bitdepth).
    '''
    wf = wave.open(fname, 'r')
    Fs = wf.getframerate()
    bd = wf.getsampwidth() * 8  # bit depth calc
    frames = wf.readframes(wf.getnframes())
    # hack to read data using array protocol type string
    dt = numpy.dtype('i' + str(wf.getsampwidth()))
    sig = numpy.fromstring(frames, dtype=dt)
    wf.close()
    return (sig, Fs, bd)
