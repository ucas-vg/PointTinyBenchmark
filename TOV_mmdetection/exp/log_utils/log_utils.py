key_map = {
    "Location eval: (AP/AR) @[ dis=1.0	| area=reasonable	| maxDets=300]": [1.0, 'reasonable', 300],
    "Location eval: (AP/AR) @[ dis=1.0	| area=small	| maxDets=300]": [1.0, 'small', 300],
    "Location eval: (AP/AR) @[ dis=1.0	| area=tiny	| maxDets=300]": [1.0, 'tiny', 300],
    "Location eval: (AP/AR) @[ dis=1.0	| area=all	| maxDets=300]": [1.0, 'all', 300]
}


def parse_log(log_file, key_map2):
    f = open(log_file)
    aps = {key: [] for key1, key in key_map2.items()}
    ars = {key: [] for key1, key in key_map2.items()}
    for line in f.readlines():
        line = line.strip('\n')
        for key1, key in key_map2.items():
            idx = line.find(key1)
            if idx != -1:
                line = line[idx+len(key1):]
                line = line.split('=')[-1]
                ap, ar = line.split('/')
                ap, ar = float(ap) * 100, float(ar) * 100
                aps[key].append(ap)
                ars[key].append(ar)
    return {"ap": aps, "ar": ars}