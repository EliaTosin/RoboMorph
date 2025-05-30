from pathlib import Path
import time
import torch
import numpy as np
import math
from functools import partial
from dataset import WHDataset, LinearDynamicalDataset
from torch.utils.data import DataLoader
from transformer_sim import Config, TSTransformer
from original.transformer_onestep import warmup_cosine_lr
import tqdm
import argparse
import wandb


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meta system identification with transformers')

    # Overall
    parser.add_argument('--model-dir', type=str, default="out", metavar='S',
                        help='Saved model folder')
    parser.add_argument('--out-file', type=str, default="ckpt", metavar='S',
                        help='Saved model name')
    parser.add_argument('--in-file', type=str, default="ckpt", metavar='S',
                        help='Loaded model name (when resuming)')
    parser.add_argument('--init-from', type=str, default="scratch", metavar='S',
                        help='Init from (scratch|resume|pretrained)')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='Seed for random number generation')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='disables CUDA training')

    # Dataset
    parser.add_argument('--nx', type=int, default=7, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--nu', type=int, default=7, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--ny', type=int, default=3, metavar='N',
                        help='model order (default: 5)')
    
    parser.add_argument('--seq-len-ctx', type=int, default=400, metavar='N',
                        help='sequence length (default: 300)')
    parser.add_argument('--seq-len-new', type=int, default=400, metavar='N',
                        help='sequence length (default: 300)')
    parser.add_argument('--mag_range', type=tuple, default=(0.5, 0.97), metavar='N',
                        help='sequence length (default: 600)')
    parser.add_argument('--phase_range', type=tuple, default=(0.0, math.pi/2), metavar='N',
                        help='sequence length (default: 600)')
    parser.add_argument('--fixed-system', action='store_true', default=False,
                        help='If True, keep the same model all the times')

    # Model
    parser.add_argument('--n-layer', type=int, default=12, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--n-head', type=int, default=4, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--n-embd', type=int, default=128, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='bias in model')

    # Training
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--max-iters', type=int, default=1_000_000, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--warmup-iters', type=int, default=10_000, metavar='N',
                        help='number of iterations (default: 1000)')
    parser.add_argument('--lr', type=float, default=6e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='D',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--eval-interval', type=int, default=100, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--eval-iters', type=int, default=50, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--fixed-lr', action='store_true', default=False,
                        help='disables CUDA training')

    # Compute
    parser.add_argument('--threads', type=int, default=10,
                        help='number of CPU threads (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda-device', type=str, default="cuda:0", metavar='S',
                        help='cuda device (default: "cuda:0")')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='disables CUDA training')
    
    # ----------------- ----------------- Edited ---------------------------------------------

    # This parameter is made to choose which dataset for training
    #  
    parser.add_argument('--custom-dataset', action='store_true', default=True,
                    help='Dataset from Isaac or simple sys')

    cfg = parser.parse_args()

    # Other settings
    cfg.beta1 = 0.9
    cfg.beta2 = 0.95

    # Derived settings
    #cfg.block_size = cfg.seq_len
    cfg.lr_decay_iters = cfg.max_iters
    cfg.min_lr = cfg.lr/10.0  #
    cfg.decay_lr = not cfg.fixed_lr
    cfg.eval_batch_size = cfg.batch_size

    # Init wandb
    if cfg.log_wandb:
        wandb.init(
            project="sysid-meta",
            #name="run1",
            # track hyperparameters and run metadata
            config=vars(cfg)
        )

    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed) # not needed? All randomness now handled with generators

    # Create out dir
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(exist_ok=True)

    # Configure compute
    torch.set_num_threads(cfg.threads)
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device_name = cfg.cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = 'cuda' if 'cuda' in device_name else 'cpu'
    # torch.set_float32_matmul_precision("high")        # disabled in local!

    if cfg.custom_dataset ==False:

        # Create data loader
        train_ds = LinearDynamicalDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, seq_len=cfg.seq_len_ctx+cfg.seq_len_new)
        #train_ds = WHDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, seq_len=cfg.seq_len_ctx+cfg.seq_len_new,
        #                     mag_range=cfg.mag_range, phase_range=cfg.phase_range,
        #                     system_seed=cfg.seed, data_seed=cfg.seed+1, fixed_system=cfg.fixed_system)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.threads)

        # if we work with a constant model we also validate with the same (thus same seed!)
        #val_ds = WHDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, seq_len=cfg.seq_len_ctx+cfg.seq_len_new,
        #                   mag_range=cfg.mag_range, phase_range=cfg.phase_range,
        #                   system_seed=cfg.seed if cfg.fixed_system else cfg.seed+2,
        #                   data_seed=cfg.seed+3, fixed_system=cfg.fixed_system)
        val_ds = LinearDynamicalDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, seq_len=cfg.seq_len_ctx+cfg.seq_len_new)
        val_dl = DataLoader(val_ds, batch_size=cfg.eval_batch_size, num_workers=cfg.threads)


    else:

        # ---------------------------------------------------------------------------------------------------

        from torch.utils.data import random_split
        import os   
        import sys

        parent_folder = os.path.join(os.getcwd(), os.pardir) 
        parent_folder = os.path.abspath(parent_folder)
        tensors_path = os.path.join(parent_folder, "isaacgym/python/examples/Franka/out_tensors/train") 
        tensors_path = os.path.abspath(tensors_path)

        list_of_available_tensors = os.listdir(tensors_path)
        print("These are the available files in "+str(tensors_path) +":\n")
        for i in range(len(list_of_available_tensors)):
            single_pt_file_path = os.path.join(tensors_path, list_of_available_tensors[i]) 
            print("--> ",list_of_available_tensors[i])

        index_test = int(np.ceil(torch.rand(1).uniform_(0,int(len(list_of_available_tensors)-1)).numpy()))
        single_pt_file_path = os.path.join(tensors_path, list_of_available_tensors[index_test]) 
        print(" \n TESTING ON: ", list_of_available_tensors[index_test])

        # loaded = torch.load('u_and_y_scratch_18u_14y.pt')
        
        loaded = torch.load(single_pt_file_path, map_location=device) #

        @ torch.no_grad()
        def loading():
            control_action_extracted = loaded['control_action']
            position_extracted = loaded['position']
            position = torch.movedim(position_extracted.to('cpu'),-2,-3)
            control_action = torch.movedim(control_action_extracted.to('cpu'),-2,-3)
            
            control_action = control_action[:1000,:,:]
            position = position[:1000,:,:]
            return control_action,position

        control_action,position = loading()
        print(control_action.shape)
        print(position.shape)
        # del control_action_extracted
        # del position_extracted
        train_dataset = torch.utils.data.TensorDataset(position, control_action)
        split_ratio = 0.8
        train_size = int(split_ratio * len(train_dataset))
        valid_size = len(train_dataset) - train_size
        train_ds, val_ds = random_split(train_dataset, [train_size, valid_size])

        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.threads)
        val_dl = DataLoader(val_ds, batch_size=cfg.eval_batch_size, num_workers=cfg.threads)
        
        cfg.ny = position.shape[2]
        cfg.nu = control_action.shape[2]
        cfg.seq_len = position.shape[1]
        cfg.block_size = cfg.seq_len

        partition = 0.5 # % of context respect prediction

        cfg.seq_len_ctx = int(partition * cfg.seq_len)
        cfg.seq_len_new = cfg.seq_len - cfg.seq_len_ctx

        cfg.in_file = f"{cfg.out_file}_nu{cfg.nu}_nu{cfg.ny}_seq_len_ctx{cfg.seq_len_ctx}_new{cfg.seq_len_new}"
        ckpt_path = model_dir / f"{cfg.in_file}.pt"
        ckpt_is_present = os.path.isfile(ckpt_path)

        if ckpt_is_present == False:
            print(f"{cfg.in_file}.pt" + " is not present --> proceeding from scratch") 
            cfg.init_from = "scratch"
        else:
            print(f"{cfg.in_file}.pt" + " is present --> proceeding from resume") 
            cfg.init_from = "resume"

# ---------------------------------------------------------------------------------------------------

    model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, n_y=cfg.ny, n_u=cfg.nu,
                      seq_len_ctx=cfg.seq_len_ctx, seq_len_new=cfg.seq_len_new,
                       bias=cfg.bias, dropout=cfg.dropout)  
    
    if cfg.init_from == "scratch":
        gptconf = Config(**model_args)
        model = TSTransformer(gptconf)
    elif cfg.init_from == "resume" or cfg.init_from == "pretrained":
        ckpt_path = model_dir / f"{cfg.in_file}.pt"
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = Config(**checkpoint["model_args"])
        model = TSTransformer(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        #unwanted_prefix = '_orig_mod.'
        #for k, v in list(state_dict.items()):
        #    if k.startswith(unwanted_prefix):
        #        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    model.to(device)

    if cfg.compile:
        model = torch.compile(model)  # requires PyTorch 2.0

    #optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)

    if cfg.init_from == "resume":
        optimizer.load_state_dict(checkpoint['optimizer'])

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        loss = 0.0
        for eval_iter, (batch_y, batch_u) in enumerate(val_dl):
            if device_type == "cuda":
                batch_y = batch_y.pin_memory().to(device, non_blocking=True)
                batch_u = batch_u.pin_memory().to(device, non_blocking=True)
            #_, loss_iter = model(batch_u, batch_y)
            
            batch_y_ctx = batch_y[:, :cfg.seq_len_ctx, :]
            batch_u_ctx = batch_u[:, :cfg.seq_len_ctx, :]
            batch_y_new = batch_y[:, cfg.seq_len_ctx:, :]
            batch_u_new = batch_u[:, cfg.seq_len_ctx:, :]
            batch_y_sim = model(batch_y_ctx, batch_u_ctx, batch_u_new)
            loss_iter = torch.nn.functional.mse_loss(batch_y_new, batch_y_sim)
            #loss_iter = torch.nn.functional.mse_loss(batch_y_new[:, 1:, :], batch_y_sim[:, :-1, :])


            loss += loss_iter.item()
            if eval_iter == cfg.eval_iters:
                break
        loss /= cfg.eval_iters
        model.train()
        return loss

    # Training loop
    LOSS_ITR = []
    LOSS_VAL = []
    loss_val = np.nan

    if cfg.init_from == "scratch" or cfg.init_from == "pretrained":
        iter_num = 0
        best_val_loss = np.inf
    elif cfg.init_from == "resume":
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint['best_val_loss']

    get_lr = partial(warmup_cosine_lr, lr=cfg.lr, min_lr=cfg.min_lr,
                     warmup_iters=cfg.warmup_iters, lr_decay_iters=cfg.lr_decay_iters)
    time_start = time.time()

    for iter_num, (batch_y, batch_u) in tqdm.tqdm(enumerate(train_dl, start=iter_num)):

        if (iter_num % cfg.eval_interval == 0) and iter_num > 0:
            loss_val = estimate_loss()
            LOSS_VAL.append(loss_val)
            print(f"\n{iter_num=} {loss_val=:.4f}\n")
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'train_time': time.time() - time_start,
                    'LOSS': LOSS_ITR,
                    'LOSS_VAL': LOSS_VAL,
                    'best_val_loss': best_val_loss,
                    'cfg': cfg,
                }
                torch.save(checkpoint, model_dir / f"{cfg.out_file}.pt")
        # determine and set the learning rate for this iteration
        if cfg.decay_lr:
            lr_iter = get_lr(iter_num)
        else:
            lr_iter = cfg.lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_iter

        if device_type == "cuda":
            batch_y = batch_y.pin_memory().to(device, non_blocking=True)
            batch_u = batch_u.pin_memory().to(device, non_blocking=True)

        batch_y_ctx = batch_y[:, :cfg.seq_len_ctx, :]
        batch_u_ctx = batch_u[:, :cfg.seq_len_ctx, :]
        batch_y_new = batch_y[:, cfg.seq_len_ctx:, :]
        batch_u_new = batch_u[:, cfg.seq_len_ctx:, :]

        batch_y_sim = model(batch_y_ctx, batch_u_ctx, batch_u_new)
        loss = torch.nn.functional.mse_loss(batch_y_new, batch_y_sim)
        #loss = torch.nn.functional.mse_loss(batch_y_new[:, 1:, :], batch_y_sim[:, :-1, :])

        LOSS_ITR.append(loss.item())
        if iter_num % 100 == 0:
            print(f"\n{iter_num=} {loss=:.4f} {loss_val=:.4f} {lr_iter=}\n")
            if cfg.log_wandb:
                wandb.log({"loss": loss, "loss_val": loss_val})

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter_num == cfg.max_iters-1:
            break

    time_loop = time.time() - time_start
    print(f"\n{time_loop=:.2f} seconds.")

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'train_time': time.time() - time_start,
        'LOSS': LOSS_ITR,
        'LOSS_VAL': LOSS_VAL,
        'best_val_loss': best_val_loss,
        'cfg': cfg,
    }
    torch.save(checkpoint, model_dir / f"{cfg.out_file}_nu{cfg.nu}_nu{cfg.ny}_seq_len_ctx{cfg.seq_len_ctx}_new{cfg.seq_len_new}.pt")

    if cfg.log_wandb:
        wandb.finish()

