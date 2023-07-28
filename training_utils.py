import os
os.environ["TORCH_HOME"] = os.path.dirname(os.getcwd())

import datetime
import random
import re
import time
import io
import warnings
import wandb
import timm
import numpy as np
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import presets
import torch
import torch.utils.data
import torchvision
import transforms
import utils
from torch.utils.data.sampler import SubsetRandomSampler
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

######################################################################


def train_one_epoch(model, criterion, ece_criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, **kwargs):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)
            # Calculate the gradients here manually
            
            ece_loss = ece_criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, args.num_classes))
        #auc = utils.auc(output, target, pos_label=kwargs['pos_label'])
        auc_dict = utils.roc_auc_score_multiclass(output, target)
        auc = sum(auc_dict.keys()) / len(auc_dict)

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(ece_loss=ece_loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["auc"].update(auc, n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

# def get_attn_params(model):
#     attn_params = []
#     num_blocks = len(model.blocks)
    
#     for i in range(num_blocks):
#         block_attn_params = []
#         block = model.blocks[i]

#         # block_attn_params.append(block.attn.qkv.weight)
#         # block_attn_params.append(block.attn.qkv.bias)
#         # block_attn_params.append(block.attn.proj.weight)
#         # block_attn_params.append(block.attn.proj.bias)

#         block_attn_params.append(block.attn)

#     attn_params.append(block_attn_params)

#     return attn_params

def get_attn_params(model):
    attn_params = []
    num_blocks = len(model.blocks)

    # Get attention parameters from each block
    for n,p in model.named_parameters():
        if 'attn' in n:
            attn_params.append(p)   

    return attn_params 

def get_fast_params(model, args):

    fast_params = []

    # Get fast parameters from each module
    if(args.tuning_method == 'tune_attention_blocks_random'):
        for name, module in model.named_modules():
            if("_fw" in module.__class__.__name__):
                for n, param in module.named_parameters():
                    if('attn' in n):
                        fast_params.append(param)

    elif(args.tuning_method == 'tune_layernorm_blocks_random'):
        for name, module in model.named_modules():
            if("_fw" in module.__class__.__name__):
                for n, param in module.named_parameters():
                    if('norm' in n):
                        fast_params.append(param)

    elif(args.tuning_method == 'tune_blocks_random'):
        for name, module in model.named_modules():
            if("_fw" in module.__class__.__name__):
                for n, param in module.named_parameters():
                    fast_params.append(param)

    return fast_params

# def get_params(model, args):
#     params = []

#     if(args.tuning_method == 'tune_attention_blocks_random'):
#         for n,p in model.named_parameters():
#             if('attn' in n):
#                 params.append(p)
        
#     elif(args.tuning_method == 'tune_layernorm_blocks_random'):
#         for n,p in model.named_parameters():
#             if('norm' in n):
#                 params.append(p)

#     elif(args.tuning_method == 'tune_blocks_random'):
#         for n,p in model.named_parameters():
#             params.append(p)

#     return params


def check_trainability(params):
    # Check if all the parameters in the 'params' list are trainable

    for p in params:
        if not p.requires_grad:
            print(p)
            return False

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img

def track_mask(mask, mask_dict):
    # mask is a torch tensor
    # mask is a list of 0s and 1s

    mask = mask.detach().cpu().numpy()
    mask = mask.tolist()

    for i in range(len(mask_dict)):
        mask_dict['mask_el_'+str(i)].append(round(mask[i], 6))

    return mask_dict

def plot_binary_mask(binary_mask):
    binary_mask = binary_mask.detach().cpu().numpy()
    heatmap_data = binary_mask.reshape((1, -1))
    fig, ax = plt.subplots(figsize=(len(binary_mask)*2, 2))
    sns.heatmap(heatmap_data, cmap='coolwarm', cbar=False, ax=ax, square=True, linecolor='black', linewidths=1)
    plt.tight_layout()
    return np.array(fig2img(fig))

def plot_mask(args, mask_dict):
    keys = mask_dict.keys()
    values = mask_dict.values()

    # Create subplots
    fig, axs = plt.subplots(len(mask_dict), 1, figsize=(8, 6), sharex=True)

    for i, (key, value) in enumerate(zip(keys, values)):
        axs[i].plot(value)
        axs[i].set_ylabel(key)
        axs[i].set_ylim(bottom=0, top=2.5, auto=True)

    # Add labels and title
    axs[-1].set_xlabel('Training Steps')
    #axs[-1].set_xticks()
    fig.suptitle('Change in Mask Values during Training')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the plot as a PNG image
    plot_path = 'Mask_plot_' + args.dataset + '_MaskGen_' + args.mask_gen_method + '.png'
    plt.savefig(os.path.join(args.fig_savepath, plot_path))

    if(args.wandb_logging):
        #wandb.log({"Mask Params during Training": fig})
        plot_path = os.path.join(args.fig_savepath, plot_path)
        # wandb.log({"Mask Params during Training": wandb.Image(plot_path,
        #                                                       caption="Mask Params during Training")})

    # for key, value in zip(keys, values):
    #     plt.plot(value, label=key)

    # plt.xlabel('Mask Parameters')
    # plt.ylabel('Mask Values')
    # plt.title('Change in Mask Values during Training')
    # plt.legend()

    # plt.savefig('Mask_plot.png')


def meta_train_one_epoch(model, criterion, ece_criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)
            ece_loss = ece_criterion(output, target)

            # Calculate the gradients here manually
            # 1. Collect attention parameters
            attn_params = get_attn_params(model)

            # 2. Calculate the gradients (manually)
            grad = torch.autograd.grad(loss, attn_params, create_graph=True)

            # 3. Update the attention parameters using the update equation
            for k, weight in enumerate(attn_params):
                if weight.fast is None:
                    attn_params[k] = weight - args.meta_lr * args.mask[k] * grad[k]
                else:
                    attn_params[k] = weight.fast - args.meta_lr * args.mask[k] * grad[k]   

            # 4. TODO: We might need to clip the mask between 0 and 1

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, args.num_classes))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(ece_loss=ece_loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

def evaluate(model, criterion, ece_criterion, data_loader, device, args, print_freq=100, log_suffix="", **kwargs):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            ece_loss = ece_criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, args.num_classes))
            #auc = utils.auc(output, target, pos_label=kwargs['pos_label'])
            # auc_dict = utils.roc_auc_score_multiclass(output, target)
            # auc = sum(auc_dict.keys()) / len(auc_dict)
            auc = 0

            # if(args.wandb_logging):

            #     keys = list(auc_dict.keys())
            #     keys = [int(key) for key in keys]
                
            #     labels = [args.class_to_idx[key] for key in keys]
            #     values = list(auc_dict.values())

            #     data = [[x,y] for (x,y) in zip(labels, values)]
            #     table = wandb.Table(data=data, columns = ["Classes", "AUC"])
            #     wandb.log({"AUC": wandb.plot.line(table, "Classes", "AUC", title="AUC Values")})



            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.update(ece_loss=ece_loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["auc"].update(auc, n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg, loss, auc

def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, testdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()

    if(args.dataset != 'CIFAR10' and args.dataset != 'CIFAR100'):
        cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)


        print("auto_augment_policy: ", auto_augment_policy)
        print("random_erase_prob: ", random_erase_prob)
        print("ra_magnitude: ", ra_magnitude)
        print("augmix_severity: ", augmix_severity)

        transform = presets.ClassificationPresetTrain(
                    crop_size=train_crop_size,
                    interpolation=interpolation,
                    auto_augment_policy=auto_augment_policy,
                    random_erase_prob=random_erase_prob,
                    ra_magnitude=ra_magnitude,
                    augmix_severity=augmix_severity,
                )
        
        if(args.dataset == 'CIFAR10'):
            dataset = torchvision.datasets.CIFAR10(root=args.dataset_basepath, train=True,
                                        download=True, transform=transform)

            num_train = len(dataset)
            indices = list(range(num_train))
            split = int(np.floor(args.val_split * num_train))

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

        elif(args.dataset == 'CIFAR100'):
            dataset = torchvision.datasets.CIFAR100(root=args.dataset_basepath, train=True,
                                        download=True, transform=transform)
            
            num_train = len(dataset)
            indices = list(range(num_train))
            split = int(np.floor(args.val_split * num_train))

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            

        else:
            # dataset = torchvision.datasets.ImageFolder(
            #     traindir,
            #     presets.ClassificationPresetTrain(
            #         crop_size=train_crop_size,
            #         interpolation=interpolation,
            #         auto_augment_policy=auto_augment_policy,
            #         random_erase_prob=random_erase_prob,
            #         ra_magnitude=ra_magnitude,
            #         augmix_severity=augmix_severity,
            #     ),
            # )

            dataset = torchvision.datasets.ImageFolder(
                traindir,
                transform,
            )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    if(args.dataset != 'CIFAR10' and args.dataset != 'CIFAR100'):
        cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_val from {cache_path}")
        dataset_val, _ = torch.load(cache_path)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
            )
        if(args.dataset == 'CIFAR10'):

            dataset_val = torch.utils.data.Subset(dataset, valid_idx)
            dataset_test = torchvision.datasets.CIFAR10(root=args.dataset_basepath, train=False,
                                       download=True, transform=preprocessing)
        elif(args.dataset == 'CIFAR100'):
            dataset_val = torch.utils.data.Subset(dataset, valid_idx)
            dataset_test = torchvision.datasets.CIFAR100(root=args.dataset_basepath, train=False,
                                       download=True, transform=preprocessing)
        else:
            dataset_val = torchvision.datasets.ImageFolder(
                valdir,
                preprocessing,
            )
            dataset_test = torchvision.datasets.ImageFolder(
                testdir,
                preprocessing,
            )

        if args.cache_dataset:
            print(f"Saving dataset_val to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_val, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        if(args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100'):
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
        else:    
            train_sampler = torch.utils.data.RandomSampler(dataset)
            #valid_sampler = torch.utils.data.SequentialSampler(dataset_val)
            valid_sampler = torch.utils.data.RandomSampler(dataset_val)

        #test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_sampler = torch.utils.data.RandomSampler(dataset_test)

    return dataset, dataset_val, dataset_test, train_sampler, valid_sampler, test_sampler

def get_optimizer(args, parameters, meta=False):
    opt_name = args.opt.lower()
    if meta:
        lr = args.outer_lr
    else:
        lr = args.lr
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    return optimizer

def get_lr_scheduler(args, optimizer):
    args.lr_scheduler = args.lr_scheduler.lower()

    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler

def get_mixup_transforms(args):
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(args.num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(args.num_classes, p=1.0, alpha=args.cutmix_alpha))

    return mixup_transforms

def get_model_ema(model_without_ddp, args):
    
    model_ema = None
    
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    return model_ema

def get_data(args):

    if(args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100'):
        train_dir = None
        val_dir = None
        test_dir = None
    else:
        if(args.dataset == 'HAM10000'):
            path = os.path.join(args.dataset_basepath, "HAM10000_dataset/")
        elif(args.dataset == 'fitzpatrick'):
            path = os.path.join(args.dataset_basepath, "Fitzpatrick/")
        elif(args.dataset == 'breastUS'):
            path = os.path.join(args.dataset_basepath, "Breast_US_dataset_split/")
        elif(args.dataset == 'retinopathy'):
            path = os.path.join(args.dataset_basepath, "Retinopathy/")
        elif(args.dataset == 'pneumonia'):
            path = os.path.join(args.dataset_basepath, "Pneumonia_Detection/")
        elif(args.dataset == 'smdg'):
            path = os.path.join(args.dataset_basepath, "SMDG/")
        else:
            raise NotImplementedError

        train_dir = os.path.join(path, "train")
        val_dir = os.path.join(path, "val")
        test_dir = os.path.join(path, "test")

    #dataset, dataset_val, train_sampler, val_sampler = load_data(train_dir, val_dir, test_dir, args)

    dataset, dataset_val, dataset_test, train_sampler, val_sampler, test_sampler = load_data(train_dir, val_dir, test_dir, args)


    return dataset, dataset_val, dataset_test, train_sampler, val_sampler, test_sampler
    
