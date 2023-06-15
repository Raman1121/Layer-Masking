import os
#os.environ["TORCH_HOME"] = "/disk/scratch2/raman/"
os.environ["TORCH_HOME"] = os.path.dirname(os.getcwd())

import datetime
import random
import re
import time
import warnings
import timm
import pandas as pd
import numpy as np
import presets
from itertools import cycle
import torch
import torch.utils.data
import torchvision
import transforms
from utils import *
from training_utils import *
from models import vision_transformer as vit
from parse_args import *
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

torch.autograd.set_detect_anomaly(True)

def main(args):
    os.makedirs(args.fig_savepath, exist_ok=True)

    #Making directory for saving checkpoints
    if args.output_dir:
        utils.mkdir(args.output_dir)
        utils.mkdir(os.path.join(args.output_dir, 'checkpoints'))

    try:
        results_df = pd.read_csv(os.path.join(args.output_dir, args.results_df))
    except:
        results_df = pd.DataFrame(columns=['Tuning Method','Train Percent','LR','Test Acc@1','Vector Path'])

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    dataset, dataset_test, train_sampler, test_sampler = get_data(args)
    args.num_classes = len(dataset.classes)
    print("DATASET: ", args.dataset)
    print("Size of training dataset: ", len(dataset))
    print("Number of classes: ", args.num_classes)

    collate_fn = None
    mixup_transforms = get_mixup_transforms(args)

    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    loaders = zip(data_loader, cycle(data_loader_test))

    print("Creating model")

    #model = utils.get_timm_model(args.model, num_classes=args.num_classes)
    model = vit.vit_base_patch16_224(pretrained=True)
    base_model = model

    print("TUNING METHOD: ", args.tuning_method)

    # Create the mask
    
    if(args.tuning_method == 'tune_attention_blocks_random'):
        mask_length = len(model.blocks)
    elif(args.tuning_method == 'tune_attention_params_random'):
        mask_length = len(model.blocks) * 4

    print("Creating mask of length: ", mask_length)
    args.mask = utils.create_random_mask(mask_length, device)
    #args.mask.to(device)

    #masking_vector = utils.get_masked_model(model, args.tuning_method, mask=list(args.mask)) # It is already asserted that mask == masking_vector

    # Disabling all parameters except attention
    for name, param in model.named_parameters():
        if('attn' not in name):
            param.requires_grad = False

    enable_module(model.head)


    trainable_params, all_param = utils.check_tunable_params(model, True)
    trainable_percentage = 100 * trainable_params / all_param

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ece_criterion = utils.ECELoss()

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )


    #Optimizer
    inner_optimizer = get_optimizer(args, parameters)
    outer_optimizer = get_optimizer(args, [args.mask])
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    #LR Scheduler
    lr_scheduler = get_lr_scheduler(args, inner_optimizer)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    #From training_utils.py
    model_ema = get_model_ema(model_without_ddp, args)
    print("model ema", model_ema)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            inner_optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            val_acc, val_loss = evaluate(model_ema, criterion, ece_criterion, data_loader_test, args=args, device=device, log_suffix="EMA")
        else:
            val_acc, val_loss = evaluate(model, criterion, ece_criterion, data_loader_test, args=args, device=device)
        return
    
    # INNER LOOP: TRAINING PROCESS HERE

    if(args.disable_training):
        print("Training Process Skipped")
    else:
        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            loaders = zip(data_loader, cycle(data_loader_test))
            print("EPOCH: ", epoch)
            print("total epochs: ", args.epochs)
            if args.distributed:
                train_sampler.set_epoch(epoch)
            #meta_train_one_epoch(model, criterion, ece_criterion, inner_optimizer, data_loader, device, epoch, args, model_ema, scaler)

            model.train()
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
            metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

            header = f"Epoch: [{epoch}]"

            # META TRAINING: INNER LOOP
            #for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            for i, ((image, target), (image_val, target_val)) in enumerate(loaders):
            
                start_time = time.time()
                image, target = image.to(device), target.to(device)
                image_val, target_val = image_val.to(device), target_val.to(device)

                #Reset the fast weights
                attn_params = get_attn_params(model)
                for k, weight in enumerate(attn_params):
                    weight.fast = None

                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    inner_optimizer.zero_grad()
                    output = model(image)
                    loss = criterion(output, target)
                    ece_loss = ece_criterion(output, target)

                    # Calculate the gradients here manually
                    # 1. Collect attention parameters
                    attn_params = get_attn_params(model)

                    # import pdb
                    # pdb.set_trace()

                    # 2. Calculate the gradients (manually)
                    grad = torch.autograd.grad(loss, attn_params, create_graph=True)

                    # 3. Update the attention parameters using the update equation
                    for k, weight in enumerate(attn_params):
                        if weight.fast is None:
                            # = weight - args.inner_lr * args.mask[k] * grad[k]
                            weight.fast = weight - args.inner_lr * args.mask[k//4] * grad[k]
                        else:
                            #attn_params[k] = weight.fast - args.inner_lr * args.mask[k] * grad[k]   
                            weight.fast = weight.fast - args.inner_lr * args.mask[k//4] * grad[k]   

                    # 4. TODO: We might need to clip the mask between 0 and 1

                    # OUTER LOOP: META TRAINING (THIS SHOULD BE ON VALIDATION DATA)
                    output = model(image_val)
                    meta_loss = criterion(output, target_val)
                    outer_optimizer.zero_grad()
                    meta_loss.backward(retain_graph=True)
                    outer_optimizer.step()

                    #Reset the fast weights
                    for k, weight in enumerate(attn_params):
                        weight.fast = None

                    # STANDARD UPDATE: Training the inner loop again for better training

                    #TODO: Apply the updated mask here
                    binary_mask = args.mask >= 1.0
                    binary_mask = binary_mask.long()
                    print("BINARY MASK: ", binary_mask)

                    for idx, block in enumerate(model.blocks):
                        if(binary_mask[idx] == 1):
                            enable_module(block.attn)
                        else:
                            disable_module(block.attn)

                    check_tunable_params(model)
                    
                    output = model(image)
                    loss = criterion(output, target)
                    inner_optimizer.zero_grad()
                    loss.backward()
                    inner_optimizer.step()

                    # Re-enabling all the attention blocks
                    # for idx, block in enumerate(model.blocks):
                    #     enable_module(block.attn)



                    print("MASK: ", args.mask)
                    # print("Clamping the mask")
                    #args.mask = torch.clamp(args.mask, 0, 1)
                    #print("MASK AFTER CLAMPING: ", args.mask)
                    print("\n")

                acc1, acc5 = utils.accuracy(output, target, topk=(1, args.num_classes))
                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item(), lr=inner_optimizer.param_groups[0]["lr"])
                metric_logger.update(ece_loss=ece_loss.item(), lr=inner_optimizer.param_groups[0]["lr"])
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

            lr_scheduler.step()
            
            if model_ema:
                val_acc, val_loss = evaluate(model_ema, criterion, ece_criterion, data_loader_test, args=args, device=device, log_suffix="EMA")
            if args.output_dir:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": inner_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                if model_ema:
                    checkpoint["model_ema"] = model_ema.state_dict()
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                #utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoints', f"model_{epoch}.pth"))7
                ckpt_path = os.path.join(args.output_dir, 'checkpoints', "meta_checkpoint_" + args.tuning_method + ".pth")
                utils.save_on_master(checkpoint, ckpt_path)

        print("Training Finished")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    # # Validation Loss using the trained model
    # val_acc, val_loss = evaluate(model, criterion, ece_criterion, data_loader_test, args=args, device=device)

    # # OUTER LOOP: Manual update of the binary mask
    # mask_grad = torch.autograd.grad(val_loss, args.mask, create_graph=True)

    # # Update the mask parameters using the update equation
    # args.mask = args.mask - args.outer_lr * mask_grad

    # optimizer_outer.zero_grad()

    # if scaler is not None:
    #     scaler.scale(loss).backward()
    #     if args.clip_grad_norm is not None:
    #         # we should unscale the gradients of optimizer's assigned params if do gradient clipping
    #         scaler.unscale_(optimizer_outer)
    #         nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    #     scaler.step(optimizer_outer)
    #     scaler.update()
    # else:
    #     loss.backward()
    #     if args.clip_grad_norm is not None:
    #         nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    #     optimizer_outer.step()

# if model_ema and i % args.model_ema_steps == 0:
#     model_ema.update_parameters(model)
#     if epoch < args.lr_warmup_epochs:
#         # Reset ema buffer to keep copying weights during warmup period
#         model_ema.n_averaged.fill_(0)

if __name__ == "__main__":

    args = get_args_parser().parse_args()
    args.output_dir = os.path.join(os.getcwd(), args.model, args.dataset)
    #args.results_df = 'Fixed_Vectors_' + args.tuning_method + '_' + args.model + '_' + str(args.lr) + '.csv'
    args.results_df = 'Fixed_Vectors_' + args.tuning_method + '_' + args.model + '.csv'

    current_wd = os.getcwd()
    args.vector_savepath = os.path.join(current_wd, 'saved_vectors', args.model, args.dataset, args.tuning_method + '_' + str(args.lr))
    args.fig_savepath = os.path.join(args.output_dir, 'plots/')
    
    if(args.masking_vector_idx is None and args.tuning_method != 'fullft'):
        args.save_flag = True
    else:
        args.save_flag = False

    main(args)