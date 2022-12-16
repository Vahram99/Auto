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


def aliases(name):
    groups = (
              ('cl', 'classifier', 'classification'),
              ('reg', 'regressor', 'regression'),
              ('n_jobs', 'num_cores', 'thread_count', 'n_cores'),
              ('verbose', 'verbosity'),
              ('random_state', 'seed', 'random_seed')
              )

    for group in groups:
        if name in group:
            return group

    return ()




