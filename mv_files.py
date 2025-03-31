import os
import sys

#substring = "count_only_cmod"
substring = "_d_model16_"

src = sys.argv[1]
dst_path = sys.argv[2]

for root,subdirs,files in os.walk(src):
    for subdir in subdirs:
        if substring in subdir:
            src_path = os.path.join(root, subdir)
            command = f"mv {src_path} {dst_path}"
            os.system(command)
            print(command)
    #for f in files:
    #    if substring in f:
    #        path = os.path.join(root, f)
    #        command = f"rm {path}"
    #        os.system(command)
    #        print(command)
