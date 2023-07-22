
log = print


def less_than(a, b):
    if hasattr(a, '__len__') and hasattr(b, '__len__'):
        assert hasattr(a, 'any') and hasattr(b, 'any'), f"{a} and {b}"
        return (a < b).any()
    return a < b


class Logger(object):
    def __init__(self):
        self.data = {}

    def max_log(self, key, new_value, log_func=log):
        cur = self.data.get(key, None)
        if cur is None or less_than(cur, new_value):
            log_func(f"[Update] {key}: {cur} -> {new_value}.")
            self.data[key] = new_value


logger = Logger()
