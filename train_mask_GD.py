import os
os.environ["TORCH_HOME"] = os.path.dirname(os.getcwd())

import datetime
import random
import wandb
from pprint import pprint
import re
import time
import warnings
import timm
import pandas as pd
import numpy as np
import presets
from PIL import Image
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

    if(args.wandb_logging):
        wandb.init(project='Dynamic Layer Masking',
                   name=args.dataset+'_'+args.model+'_'+args.tuning_method+'_MaskGen_'+args.mask_gen_method)
        
        hparams_table = wandb.Table(dataframe=args.hparams_df, allow_mixed_types=True)
        wandb.log({"Hyper Parameters": hparams_table})


    os.makedirs(args.fig_savepath, exist_ok=True)
    track_trainable_params = []

    #Making directory for saving checkpoints
    if args.output_dir:
        utils.mkdir(args.output_dir)
        utils.mkdir(os.path.join(args.output_dir, 'checkpoints'))

    try:
        results_df = pd.read_csv(os.path.join(args.output_dir, args.results_df))
        test_results_df = pd.read_csv(os.path.join(args.output_dir, args.test_results_df))
    except:
        results_df = pd.DataFrame(columns=['Tuning Method','Train Percent','LR','Test Acc@1','Vector Path'])
        test_results_df = pd.DataFrame(columns=['Tuning Method','Train Percent','LR Scaler', 'Inner LR', 'Outer LR','Test Acc@1','Vector Path'])

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    dataset, dataset_val, dataset_test, train_sampler, val_sampler, test_sampler = get_data(args)
    args.num_classes = len(dataset.classes)
    print("DATASET: ", args.dataset)
    print("Size of training dataset: ", len(dataset))
    print("Size of validation dataset: ", len(dataset_val))
    print("Size of test dataset: ", len(dataset_test))
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
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers, pin_memory=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    loaders = zip(data_loader, cycle(data_loader_val))

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
    args.mask = utils.create_random_mask(mask_length, args.mask_gen_method, device, sigma=args.sigma)
    print("Initial Mask: ", args.mask)

    keys = ['mask_el_'+str(i) for i in range(mask_length)]
    values = [[] for i in range(mask_length)]
    MASK_DICT = {key: value for key, value in zip(keys, values)} #A dictionary to store the values of each mask param during training
    #BINARY_MASK_DICT = {key: value for key, value in zip(keys, values)} #A dictionary to store the values of each binary mask element during training
    BINARY_MASK_PLOT_ARRAYS = []

    # Track the original mask and binary mask
    MASK_DICT = track_mask(args.mask, MASK_DICT)
    binary_mask = args.mask >= 1.0
    binary_mask = binary_mask.long()
    #BINARY_MASK_DICT = track_binary_mask(binary_mask, BINARY_MASK_DICT)
    binary_mask_fig_arr = plot_binary_mask(binary_mask)
    BINARY_MASK_PLOT_ARRAYS.append(binary_mask_fig_arr)

    # Disabling all parameters except attention
    for name, param in model.named_parameters():
        if('attn' not in name):
            param.requires_grad = False

    enable_module(model.head)


    trainable_params, all_param = utils.check_tunable_params(model, True)
    trainable_percentage = 100 * trainable_params / all_param

    # Track the original trainable percentage
    if(args.wandb_logging):
        wandb.log({"Trainable Percentage": trainable_percentage})

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
    outer_optimizer = get_optimizer(args, [args.mask], meta=True)
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
            test_acc, test_loss = evaluate(model_ema, criterion, ece_criterion, data_loader_test, args=args, device=device, log_suffix="EMA")
        else:
            test_acc, test_loss = evaluate(model, criterion, ece_criterion, data_loader_test, args=args, device=device)
        return
    
    # INNER LOOP: TRAINING PROCESS HERE

    if(args.disable_training):
        print("Training Process Skipped")
    else:
        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            loaders = zip(data_loader, cycle(data_loader_val))
            print("Epoch: ", epoch)
            print("Total Epochs: ", args.epochs)
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
                    # print("Max and min values", torch.max(nn.Softmax(dim=1)(output)), torch.min(nn.Softmax(dim=1)(output)))
                    loss = criterion(output, target)
                    ece_loss = ece_criterion(output, target)

                    # Calculate the gradients here manually
                    # 1. Collect attention parameters
                    #print("Attention parameters", attn_params)
                    attn_params = get_attn_params(model)
                    #print("All params trainable" if not check_trainability(attn_params) else "Not all params trainable")
                    print("Loss: ", loss)


                    # 2. Calculate the gradients (manually)
                    try:
                        grad = torch.autograd.grad(loss, attn_params, create_graph=True)
                    except:
                        import pdb
                        pdb.set_trace()
                    inner_lr = lr_scheduler.get_last_lr()[-1]

                    if(args.wandb_logging):
                        wandb.log({"Inner LR": inner_lr})
                        wandb.log({"Outer LR": args.outer_lr})

                    # 3. Update the attention parameters using the update equation
                    for k, weight in enumerate(attn_params):
                        if weight.fast is None:
                            weight.fast = weight - inner_lr * args.lr_scaler * args.mask[k//4] * grad[k]
                        else:
                            #attn_params[k] = weight.fast - args.inner_lr * args.mask[k] * grad[k]   
                            weight.fast = weight.fast - inner_lr * args.lr_scaler * args.mask[k//4] * grad[k]   

                    # 4. TODO: We might need to clip the mask between 0 and 1

                    # OUTER LOOP: META TRAINING (THIS SHOULD BE ON VALIDATION DATA) - TRAINING THE META PARAMS (MASK)
                    output = model(image_val)
                    meta_loss = criterion(output, target_val)

                    if(args.wandb_logging):
                        wandb.log({"Meta Loss": meta_loss})

                    outer_optimizer.zero_grad()
                    meta_loss.backward(retain_graph=True)
                    outer_optimizer.step()

                    #Reset the fast weights
                    for k, weight in enumerate(attn_params):
                        weight.fast = None

                    # STANDARD UPDATE: Training the inner loop again for better training

                    #TODO: THRESHOLD THE MASK HERE
                    # TODO: CAN USE DIFFERENT SCHEMES: 1. SIGMOID
                    
                    binary_mask = args.mask >= 1.0
                    binary_mask = binary_mask.long()

                    if(args.wandb_logging):
                        binary_mask_fig_arr = plot_binary_mask(binary_mask)
                        BINARY_MASK_PLOT_ARRAYS.append(binary_mask_fig_arr)

                    ## APPLY THE UPDATED MASK
                    for idx, block in enumerate(model.blocks):
                        if(binary_mask[idx] == 1):
                            enable_module(block.attn)
                        else:
                            disable_module(block.attn)

                    trainable_params, all_param = check_tunable_params(model, False)
                    trainable_percentage = 100 * trainable_params / all_param
                    track_trainable_params.append(trainable_percentage)

                    if(args.wandb_logging):
                        wandb.log({"Trainable Percentage": track_trainable_params[-1]})
                    
                    # STANDARD UPDATE
                    print("STANDARD UPDATE")
                    output = model(image)
                    loss = criterion(output, target)
                    inner_optimizer.zero_grad()
                    loss.backward()
                    inner_optimizer.step()

                    acc1, acc5 = utils.accuracy(output, target, topk=(1, args.num_classes))
                    print("ACC1: {}, ACC5: {}, LOSS: {}".format(acc1, acc5, loss))

                    if(args.wandb_logging):
                        wandb.log({"Train Accuracy": acc1})
                        wandb.log({"Standard Update Loss": loss})

                    # Re-enabling all the attention blocks
                    for idx, block in enumerate(model.blocks):
                        enable_module(block.attn)

                    print("MASK: ", args.mask)
                    MASK_DICT = track_mask(args.mask, MASK_DICT)
                    #BINARY_MASK_DICT = track_mask(binary_mask, BINARY_MASK_DICT)

                    if(args.wandb_logging):
                        temp_binary_mask = binary_mask.cpu().detach().numpy().tolist()

                        _df_mask = pd.DataFrame(MASK_DICT)

                        table = wandb.Table(dataframe=_df_mask)
                        #binary_table = wandb.Table(dataframe=_df_binary_mask)

                        #wandb.log({"Mask Params": table, "Binary Mask Params": binary_table})
                        wandb.log({"Mask Params": table})
                    
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
                val_acc, val_loss = evaluate(model_ema, criterion, ece_criterion, data_loader_val, args=args, device=device, log_suffix="EMA")
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

        if(args.wandb_logging):
            BINARY_MASK_PLOT_ARRAYS = np.vstack(BINARY_MASK_PLOT_ARRAYS)
            plot_img = Image.fromarray(BINARY_MASK_PLOT_ARRAYS)
            wandb.log({"Binary Mask Plot": wandb.Image(plot_img)})


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        # Plotting the change in mask during training
        plot_mask(args, MASK_DICT)

        val_acc, val_loss = evaluate(model, criterion, ece_criterion, data_loader_val, args=args, device=device)
        print("Val accuracy: ", val_acc)
        print("Val loss: ", val_loss)

        test_acc, test_loss = evaluate(model, criterion, ece_criterion, data_loader_test, args=args, device=device)
        print("Test accuracy: ", test_acc)
        print("Test loss: ", test_loss)

        #columns=['Tuning Method','Train Percent','LR Scaler', 'Inner LR', 'Outer LR','Test Acc@1','Vector Path']
        new_row = ['Dynamic_'+args.tuning_method, np.mean(track_trainable_params), args.lr_scaler, args.lr, args.outer_lr, test_acc, np.nan]
        test_results_df.loc[len(test_results_df)] = new_row
        test_results_df.to_csv(os.path.join(args.output_dir, args.test_results_df), index=False)
    

if __name__ == "__main__":

    args = get_args_parser().parse_args()
    args.output_dir = os.path.join(os.getcwd(), args.model, args.dataset)
    #args.results_df = 'Fixed_Vectors_' + args.tuning_method + '_' + args.model + '_' + str(args.lr) + '.csv'
    args.results_df = 'Fixed_Vectors_' + args.tuning_method + '_' + args.model + '.csv'
    args.test_results_df = 'Test_set_results_' + args.tuning_method + '_' + args.model + '.csv'
    current_wd = os.getcwd()
    args.vector_savepath = os.path.join(current_wd, 'saved_vectors', args.model, args.dataset, args.tuning_method + '_' + str(args.lr))
    args.fig_savepath = os.path.join(args.output_dir, 'plots/')
    
    if(args.masking_vector_idx is None and args.tuning_method != 'fullft'):
        args.save_flag = True
    else:
        args.save_flag = False

    args.val_split = 0.2

    hparams_df = pd.DataFrame(columns=['Hparams', 'Value'])
    hparams_df.loc[len(hparams_df)] = ['Inner LR', args.lr]
    hparams_df.loc[len(hparams_df)] = ['Outer LR', args.outer_lr]
    hparams_df.loc[len(hparams_df)] = ['LR Scaler', args.lr_scaler]
    hparams_df.loc[len(hparams_df)] = ['Tuning Method', str(args.tuning_method)]
    hparams_df.loc[len(hparams_df)] = ['LR Warmup Epochs', args.lr_warmup_epochs]
    hparams_df.loc[len(hparams_df)] = ['Sigma', args.sigma]
    hparams_df.loc[len(hparams_df)] = ['Mask Gen Method', args.mask_gen_method]
    hparams_df.loc[len(hparams_df)] = ['Dataset', args.dataset]

    print(hparams_df)
                                    
    args.hparams_df = hparams_df

    main(args)