#!/usr/bin/env python3
from collections import OrderedDict, deque

'''
handy data structure related items
'''
# select values from an itr by index
def itr_iselect(itr, *indices):
    i_set = set(indices)
    return (e for (i, e) in enumerate(itr) if i in i_set)

def print_table(itr, field_header=['signal files'], delim='|'):
    '''pretty print iterable in a column'''
    itr = [i for i in itr]
    max_width = max(len(field) for field in itr)
    widths = iter(lambda:max_width, 1) # an infinite width generator

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

# like it sounds : an ordered, int subscriptable dict (OID)
class OrderedIndexedDict(OrderedDict):
    def __getitem__(self, key):
        #FIXME: how to handle slices?
        #(it's weird since we always have to check first and using the list...)
        if isinstance(key, slice ):
            print("you've passed a slice! with start",key.start,"and stop",key.stop)
            return list(self.values())[key]

        # if it's already mapped get the value
        elif key in self:
            return OrderedDict.__getitem__(self, key)

        # FIXME: is this the fastest way to implement this?
        # if it's an int, iterate the linked list and return the value
        elif isinstance(key, int):
            return self[self._get_key(key)]
            # return self._setget_value(key)

    def __setitem__(self, key, value):
        # don't give me ints bitch...(unless you're changing a value)
        if isinstance(key, int):# raise KeyError("key can not be of type integer")
            key = self._get_key(key)
        OrderedDict.__setitem__(self, key, value)

    def _get_key(self, key):
        ''' get the key for a given index'''
        # check for out of bounds
        l = len(self)
        if l - 1 < key or key < -l :
            raise IndexError("index out of range, len is " + str(l))

        # get the root of the doubly linked list (see the OrderedDict implemenation)
        root = self._OrderedDict__root
        curr = root.next
        act = lambda link : link.next

        # handle -ve indices too
        if key < 0 :
            key += 1
            curr = root.prev
            act = lambda link : link.prev
        # traverse the linked list for our element
        for i in range(abs(key)):
            curr = act(curr)
        # return the key
        return curr.key

# a handy alias ... with a sloppy ending
Lict = OrderedIndexedDict

'''
mpl convenience
'''
def label_ymax(axis, x, label):
    '''add an optional label to the line @ ymax'''
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
    '''
    use max value from the available lines for annotation placement
    '''
    lines = axes.get_lines()
    mx = 0
    for line in lines:
        lm = max(line.get_ydata())
        if lm > mx:
            mx = lm
    return mx
