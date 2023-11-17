import torch
import numpy as np
import time
from tqdm import tqdm
import os
import sys
import accelerate

from dl_utils.save_io import save_checkpt, load_json_or_yaml
from dl_utils.datas import get_datasets
from dl_utils.training import record_session
from dl_utils.utils import package_versions
from dl_utils.seq_models import make_model
from dl_utils.schedulers import DecayScheduler

"""
To use this script, move it one level above the dl_utils directory.

This script runs a toy sequence training to ensure that your model
classes are working. The sequence is a starting number that can take
k possible forms, a string of N ordered digits ranging somewhere between
1-100, and a final output of the starting number.
"""

def train(rank, config, verbose=True, *args, **kwargs):
    torch.cuda.empty_cache()

    # Hyperparameters
    config = config_error_catching(config) # Make sure we have valid config
    config["packages"] = package_versions()
    config["seed"] = config.get("seed", int(time.time()))
    if config["seed"] is None: config["seed"] = int(time.time())
    torch.manual_seed(config["seed"]+rank)
    np.random.seed(   config["seed"]+rank)
    config["rank"] = rank

    # Dataset/Tokenizer
    #######################################
    if verbose and rank==0: print("Making Data")
    # This function updates the config dict and returns DataSet objects
    tokenizer, train_dataset, val_dataset = get_datasets(config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.get("batch_size", 128)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=config.get("batch_size", 1000)
    )
    if verbose and rank==0:
        print("Train Samples:", len(train_dataset))
        print("Val Samples:", len(val_dataset))
        print("Using Sequence Length:", config["seq_len"])

    # Model
    #######################################
    model = make_model(config)
    n_params = 0
    for p in model.parameters():
        if hasattr(p, "data"):
            n_params += p.data.numel()
    config["n_params"] = n_params
    print("NParameters:", n_params)

    # Optimizer
    #######################################
    if verbose and rank==0:
        print("Creating Optimizer")
    config["lr"] = config.get("lr", 0.001)
    optimizer = getattr(torch.optim, config.get("optim_type","Adam"))(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("l2", 0),
    )

    # Scheduler
    #######################################
    scheduler = DecayScheduler( optimizer, **config )

    # Distributed Wrapper
    #######################################
    if rank==0 and verbose and torch.cuda.device_count()>1:
        print("Handling multiple GPUs")
    accelerator = accelerate.Accelerator()
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    val_loader = accelerator.prepare(val_loader)

    #############################################################
    # Save Configuration
    #############################################################
    if config.get("save", False):
        record_session(config, model)


    #############################################################
    # Training
    #############################################################
    n_epochs = config.get("n_epochs", 100)
    for epoch in range(n_epochs):
        epochtime = time.time()
        torch.cuda.empty_cache()
        if rank==0 and verbose:
            print()
            s = "Beginning Epoch {} - {}".format(
                epoch, config.get("save_folder", "No Save Folder")
            )
            print(s)
            logstr = s + "\n"

        #############################################################
        # Train Loop
        #############################################################
        model.train()
        avg_loss = 0
        avg_acc = 0
        nloops = config.get("n_train_loops", len(train_loader))
        nloops = min(nloops,len(train_loader))
        checkpt_mod = config.get( "checkpt_mod", np.inf )
        val_mod = config.get( "val_mod", 1)
        optimizer.zero_grad()
        for i,data in enumerate(train_loader):
            starttime = time.time()
            package = model(
                data,
                ret_preds=True,
                tforce=config.get("tforce_train", True),
            )
            loss = package["loss"]
            acc = package["acc"]

            accelerator.backward(loss)

            avg_acc += acc.item()
            avg_loss += loss.item()

            if i%config.get("n_grad_loops",1)==0 or i==len(train_loader)-1:
                if config.get("grad_clip",0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["grad_clip"]
                    )
                optimizer.step()
                optimizer.zero_grad()
                try:
                    scheduler.step()
                except:
                    pass

            if verbose and i%10==0 and rank==0:
                dec = 4
                l = round(loss.item(), dec)
                a = round(acc.item(), dec)
                c = round(100*i/nloops, 2)
                t = round(time.time()-starttime, 3)
                s = "Loss: {} -Acc: {}".format(l,a)
                s += " - {}% {}s   ".format(c,t)
                print(s, end=int(len(s)/2)*" " + "\r")


            if config.get("exp_name","deleteme")=="test" and i>=30: break
            if i>=(nloops-1): break
        div = (i+1)
        dec = 5
        train_loss = round(avg_loss/div, dec)
        train_acc  = round(avg_acc/div, dec)
        if verbose:
            s = "Example Train Preds:"
            print()
            print(s)
            logstr += s+"\n"
            preds = package["pred_ids"]
            targs = data["output_ids"]
            for i in range(min(3,len(preds))):
                s = "Targ: "+", ".join(
                    [str(t) for t in targs[i].cpu().data.tolist()]
                )
                logstr += s+"\n"
                print(s)
                s = "Pred: "+", ".join(
                    [str(p) for p in preds[i].cpu().data.tolist()]
                )
                logstr += s+"\n\n"
                print(s)
                print()

        #############################################################
        # Validation Loop
        #############################################################
        val_loss =     0
        val_acc =      0
        if rank==0 and (epoch%val_mod==0 or epoch==n_epochs-1):
            model.eval()
            if verbose: print("Validating...")
            with torch.no_grad():
                nloops = config.get("max_val_loops",len(val_loader))
                nloops = min(nloops, len(val_loader))
                avg_loss = 0
                avg_acc = 0
                for i,data in enumerate(val_loader):
                    starttime = time.time()
                    package = model(
                        data,
                        ret_preds=True,
                        tforce=False,
                        temperature=config.get(
                            "sampling_temperature", None
                        )
                    )
                    loss = package["loss"]
                    acc = package["acc"]
                    avg_loss += loss.item()
                    avg_acc += acc.item()
                    if verbose:
                        p = round(100*(i+1)/nloops, 2)
                        t = round(time.time()-starttime, 4)
                        print("{}% -- {}s".format(p,t), end="         \r")
                    if i>=nloops-l: break
            div = (i+1)
            dec = 5
            val_loss = round(avg_loss/div, 5)
            val_acc =  round(avg_acc/div, 5)
            scheduler.step(val_loss)
            if config.get("exp_name", "deleteme")=="test": break
            if verbose:
                print()
                s = "Example Val Preds:"
                print(s)
                logstr += s+"\n"
                preds = package["pred_ids"]
                targs = data["output_ids"]
                for i in range(min(3,len(preds))):
                    s = "Targ: "+", ".join(
                        [str(t) for t in targs[i].cpu().data.tolist()]
                    )
                    logstr += s+"\n"
                    print(s)
                    s = "Pred: "+", ".join(
                        [str(p) for p in preds[i].cpu().data.tolist()]
                    )
                    logstr += s+"\n"
                    print(s)
                    print()
                print()
                s = "Final Stats, Epoch: {}".format(epoch)
                print(s)
                logstr += "\n" + s + "\n"

                s = "Train Loss: {} - Train Acc: {}".format(
                    train_loss,train_acc
                )
                logstr += s + "\n"
                print(s)

                s = "Val Loss: {} Val Acc: {}".format( val_loss,val_acc)
                logstr += s + "\n"
                print(s)

                s = "Epoch Dur: {}s".format(round(time.time()-epochtime))
                logstr += s + "\n\n\n\n"
                print(s)

                print()
                print()

        ##############################################################
        #### SAVING
        ##############################################################
        if rank==0 and epoch%val_mod==0 and config.get("save", False):
                save_dict = {
                    "mid_epoch": False,
                    "epoch":       epoch,
                    "train_loss":  train_loss,
                    "train_acc":   train_acc,
                    "val_loss":    val_loss,
                    "val_acc":     val_acc,
                    "state_dict":  model.state_dict(),
                    "optim_dict":  optimizer.state_dict(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "config":        config,
                }
                mod = config.get("sd_save_mod", None)
                keep_prev_sd = mod and epoch%mod==0
                save_checkpt(
                    save_dict=save_dict,
                    save_folder=config["save_folder"],
                    save_name="checkpt",
                    epoch=epoch,
                    ext=".pt",
                    del_prev_sd=not keep_prev_sd
                )
                save_training_log(config, logstr)

        # Clean up
        keys = list(package.keys())
        for k in keys: del package[k]
        if config.get("exp_name", "deleteme")=="test" and epoch>2: break
    return model


def save_training_log(config, logstr, fname="training_log.txt", reset=False):
    """
    Saves the logstr to the save folder under the name training_log.txt

    config: dict
    logstr: str
        the string to save
    fname: str
        the name of the file to save to
    reset: bool
        if true, resets the training log and then writes. otherwise
        appends to training log
    """
    mode = "w" if reset else "a"
    with open(os.path.join(config["save_folder"], fname),mode) as f:
        f.write(logstr)

def config_error_catching(config):
    """
    This function just makes sure that some obvious hyperparameter
    choices are set and some obviously wrong hyperparameter settings
    are changed to what the experimenter meant.
    """
    return config

if __name__=="__main__":
    config = { }
    if len(sys.argv)>1:
        config = load_json_or_yaml(sys.argv[1])
    train(0, config)

