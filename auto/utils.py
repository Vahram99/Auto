import inspect


def disable(method):
    def inner(self, *_, **__):
        raise AttributeError("'{}' object has no attribute '{}'"
                             .format(type(self).__name__, method.__name__))
    return inner


def hasarg(func, arg):
    sig = inspect.signature(func)
    if arg in sig.parameters:
        return True
    return False




