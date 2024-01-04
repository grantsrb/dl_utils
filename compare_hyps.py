"""
Argue two JSON files. This script will print out the values of all keys that
are different between the two json files.

    $ python3 compare_hyps.py file0.json file1.json

Prints:
    my_key:  value0 -- value1
    my_key2: value0 -- value1
"""

import json
import sys
import os
if sum([int(f=="dl_utils") for f in os.listdir("./")])==0:
    sys.path.append("../")
else:
    sys.path.append("./")
import dl_utils

if __name__=="__main__":
    data_list = []
    for arg in sys.argv[1:]:
        if ".json" in arg or ".yaml" in arg:
            data_list.append(dl_utils.save_io.load_json_or_yaml(arg))

    keys = set()
    for d in data_list:
        keys = keys.union(set(d.keys()))
    keys = sorted(list(keys))

    for k in keys:
        all_equal = True
        for i,data in enumerate(data_list):
            if k not in data: data[k] = "UNDEFINED"
            if i == 0: val = data[k]
            elif val != data[k]: all_equal = False
        if not all_equal:
            s = str(k) + ": "
            for data in data_list:
                s += str(data[k]) + " --- "
            s = s[:-4]
            print(s)

