'''
Plotter markup language
'''
import np
import inspect
import wrapt
import functools
from utils import log

_uses = {}


def plot(sys=None, fig=None, loc=(1, 1)):
    '''Plot the output of this system in the simulation pane
    '''
    if sys is None:  # case where extra args were passed
        return functools.partial(plot, fig=fig, loc=loc)

    # register sys in uses table
    nonlocal _uses
    contexts = _uses.setdefault(sys, {})

    @wrapt.decorator
    def capture_output(sys, instance, args, kwargs):
        try:
            frame = inspect.currentframe()
            tb = inspect.getframeinfo(sys)
            log.info("system '{}' found in context '{code_context}' line "
                     "'{lineno}'".format(sys.__name__, **tb))
            # call system and append result to tracking array
            out = sys(*args, **kwargs)
            contexts.setdefault(tb.code_context,
                np.zeros(sys._context.max_buf_len)
            )[:] = out
            # np.append(contexts.setdefault(tb.code_context, np.array([])), out)

        finally:
            del frame
        return out

    return capture_output(sys)
