import os
import sys

#substring = "count_only_cmod"
substring = "_d_model8_"

for arg in sys.argv[1:]:
    for root,subdirs,files in os.walk(arg):
        for subdir in subdirs:
            if substring in subdir:
                path = os.path.join(root, subdir)
                command = f"rm -rf {path}"
                os.system(command)
                print(command)
        #for f in files:
        #    if substring in f:
        #        path = os.path.join(root, f)
        #        command = f"rm {path}"
        #        os.system(command)
        #        print(command)
