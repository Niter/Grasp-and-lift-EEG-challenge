import pdb

_breakpoints = {}

def reset_breakpoints(disabled=[]):
    global _breakpoints
    _breakpoints = dict((x, False) for x in disabled)

def set_breakpoint(tag, condition=True):
    if tag not in _breakpoints:
        _breakpoints[tag] = True
        if condition:
            pdb.set_trace()
    else:
        if _breakpoints[tag] and condition:
            pdb.set_trace()

# Example
def mycode():
    some_command()
    set_breakpoint('mycode0')
    another_command()
    set_breakpoint('mycode1', x == 4)

