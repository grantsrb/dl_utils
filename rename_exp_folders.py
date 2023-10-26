import torch
import os
import sys
import shutil

import dl_utils.save_io as sio

"""
Use this script to rename a number of model folders to a new name.
"""

if __name__=="__main__":
    model_folders = []
    output_dir = None
    for i,arg in enumerate(sys.argv[1:]):
        if "--o=" in arg or "--output_dir=" in arg:
            output_dir = os.path.abspath(os.path.expanduser(arg.split("=")[-1]))
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        elif i==0 or (i==1 and output_dir is not None):
            new_exp_name = sys.argv[1]
        else:
            full_path = os.path.abspath(os.path.expanduser(arg))
            if sio.is_model_folder(arg):
                model_folders.append(full_path)
            else:
                for mf in sio.get_model_folders(full_path,incl_full_path=True):
                    model_folders.append(mf)

    for mf in model_folders:
        print("Folder:", mf)
        if mf[-1] == "/": mf = mf[:-1]
        splt = mf.split("/")
        if len(splt) > 1:
            root_path = "/".join(splt[:-1])
            splt = splt[-1].split("_")
        else:
            root_path = "./"
            splt = splt[0].split("_")
        for i,s in enumerate(reversed(splt[1:])):
            try:
                exp_num = int(s)
                idx = -(i+1)
            except:
                pass
        current_name = "_".join(splt[:idx])
        end_path = "_".join(splt[idx:])
        if current_name==new_exp_name: continue
        new_dir = new_exp_name+"_"+end_path
        if output_dir is not None:
            root_path = output_dir
        new_full = os.path.join(root_path, new_dir)
        if os.path.exists(new_full):
            print("Path Already Exists for", new_full)
            continue
        for checkpt in sio.get_checkpoints(mf):
            pt = sio.load_checkpoint(checkpt)
            pt["hyps"]["exp_name"] = new_exp_name
            pt["hyps"]["save_folder"] = new_full
            pt["hyps"]["model_folder"] = new_dir
            pt["hyps"]["exp_folder"] = root_path
            torch.save(pt, checkpt)

        if not os.path.exists(root_path):
            os.makedirs(root_path)
        os.rename(mf, new_full)
        print("Success!")
        print("New Folder:", new_full)
        print()
