def remove_space(s):
    return s.replace(' ', '').replace('\t', '')


key_map = {
    remove_space("Location eval: (AP/AR) @[ dis=1.0	| area=reasonable	| maxDets=300]"): [1.0, 'reasonable', 300],
    remove_space("Location eval: (AP/AR) @[ dis=1.0	| area=small	| maxDets=300]"): [1.0, 'small', 300],
    remove_space("Location eval: (AP/AR) @[ dis=1.0	| area=tiny	| maxDets=300]"): [1.0, 'tiny', 300],
    remove_space("Location eval: (AP/AR) @[ dis=1.0	| area=all	| maxDets=300]"): [1.0, 'all', 300],
    remove_space("Location eval: (AP/AR) @[ dis=1.0       | area=all      | maxDets=100]"): [1.0, 'all', 100],
    remove_space("Location eval: (AP/AR) @[ dis=1.0       | area=tiny      | maxDets=100]"): [1.0, 'tiny', 100],
    remove_space("Location eval: (AP/AR) @[ dis=1.0       | area=small      | maxDets=100]"): [1.0, 'small', 100],
    remove_space("Location eval: (AP/AR) @[ dis=1.0       | area=reasonable      | maxDets=100]"): [1.0, 'reasonable', 100],
}

key_map2 = {}
for key1, key in key_map.items():
    key_map2[key1] = "_".join([str(e) for e in key])


def got_class_name(line):
    idx = line.find(')')
    if idx != -1:
        return line[1:idx]
    return ""


def parse_log(log_file):
    lines = open(log_file).readlines() if isinstance(log_file, str) else log_file
    aps = {key: [] for key1, key in key_map2.items()}
    ars = {key: [] for key1, key in key_map2.items()}
    aps_data = {k: {key: [] for key1, key in key_map2.items()} for k in ['class_name']}
    ars_data = {k: {key: [] for key1, key in key_map2.items()} for k in ['class_name']}
    for line in lines:
        line = line.strip('\n')
        for key1, key in key_map2.items():
            idx = remove_space(line).find(key1)
            if idx != -1:
                class_name = got_class_name(line)

                line = line[idx+len(key1):]
                line = line.split('=')[-1]
                ap, ar = line.split('/')
                ap, ar = float(ap) * 100, float(ar) * 100
                aps[key].append(ap)
                ars[key].append(ar)
                aps_data['class_name'][key].append(class_name)
                ars_data['class_name'][key].append(class_name)
    return {"ap": aps, "ar": ars, "ap_data": aps_data, "ar_data": ars_data}


import matplotlib.pyplot as plt

def vis_res(res):
    aps, ars = res["ap"], res["ar"]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for key, d in aps.items():
        plt.plot(range(1, len(d)+1), d, label=key)
    plt.xlabel("epoch")
    plt.ylabel("AP")
    plt.legend()

    plt.subplot(1, 2, 2)
    for key ,d, in ars.items():
        plt.plot(range(1, len(d)+1), d, '--', label=key)
    plt.xlabel("epoch")
    plt.ylabel("MAX AR")
    plt.legend()
    plt.show()