class InputArgs(object):
    def __init__(self, args, ignore_first=False):
        if ignore_first:
            self.is_first = True
        else:
            self.is_first = False
        self.args = args

    def update_from_input(self):
        if self.is_first:
            self.is_first = False
            return True
        r = input("update args([arg_name=arg_value,], 'q' for quit): \n")
        if len(r) == 0 or r == 'q':
            return False
        for e in r.split(','):
            e = e.strip()
            if len(e) == 0:
                continue
            name, value = e.split('=')
            name = name.strip()
            value = value.strip()
            if name in self.args:
                self.args[name] = type(self.args[name])(value)
        return self.args

    def __getitem__(self, item):
        return self.args[item]


def input_yes(info):
    x = input(info)
    if x.lower() in ['y', 'yes']:
        return True
    return False
