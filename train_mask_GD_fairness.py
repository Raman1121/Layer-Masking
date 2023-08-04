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
import yaml

torch.autograd.set_detect_anomaly(True)


def main(args):
    if args.wandb_logging:
        wandb.init(
            project="Dynamic Layer Masking",
            name=args.dataset
            + "_"
            + args.model
            + "_"
            + args.tuning_method
            + "_MaskGen_"
            + args.mask_gen_method,
        )

        hparams_table = wandb.Table(dataframe=args.hparams_df, allow_mixed_types=True)
        wandb.log({"Hyper Parameters": hparams_table})

    os.makedirs(args.fig_savepath, exist_ok=True)
    track_trainable_params = []

    # Making directory for saving checkpoints
    if args.output_dir:
        utils.mkdir(args.output_dir)
        utils.mkdir(os.path.join(args.output_dir, "checkpoints"))

    try:
        results_df = pd.read_csv(os.path.join(args.output_dir, args.results_df))
        test_results_df = pd.read_csv(
            os.path.join(args.output_dir, args.test_results_df)
        )
    except:
        results_df = pd.DataFrame(
            columns=[
                "Tuning Method",
                "Train Percent",
                "LR",
                "Test Acc@1",
                "Vector Path",
            ]
        )
        if(args.sens_attribute == 'gender'):
            test_results_df = pd.DataFrame(
                    columns=[
                        "Tuning Method",
                        "Train Percent",
                        "LR Scaler",
                        "Inner LR",
                        "Outer LR",
                        "Test Acc@1",
                        "Test Acc Male",
                        "Test Acc Female",
                        "Test Acc Difference",
                        "Vector Path",
                    ]
                )
        elif(args.sens_attribute == 'skin_type' or args.sens_attribute == 'age'):
            test_results_df = pd.DataFrame(
                columns=[
                    "Tuning Method",
                    "Train Percent",
                    "LR Scaler",
                    "Inner LR",
                    "Outer LR",
                    "Test Acc@1",
                    "Test Acc (Best)",
                    "Test Acc (Worst)",
                    "Test Acc Difference",
                    "Vector Path",
                ]
            ) 
        else:
            raise NotImplementedError

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    with open('config.yaml') as file:
        yaml_data = yaml.safe_load(file)

    (
        dataset,
        dataset_val,
        dataset_test,
        train_sampler,
        val_sampler,
        test_sampler,
    ) = get_fairness_data(args, yaml_data)

    args.num_classes = len(dataset.classes)
    print("DATASET: ", args.dataset)
    print("Size of training dataset: ", len(dataset))
    print("Size of validation dataset: ", len(dataset_val))
    print("Size of test dataset: ", len(dataset_test))
    print("Number of classes: ", args.num_classes)
    pprint(dataset.class_to_idx)

    #args.class_to_idx = {value: key for key, value in dataset.class_to_idx.items()}
    # print(args.class_to_idx[0])
    # print(type(args.class_to_idx.keys()[0]))

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
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        #drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    loaders = zip(data_loader, cycle(data_loader_val))

    print("Creating model")

    # model = utils.get_timm_model(args.model, num_classes=args.num_classes)
    model = vit.vit_base_patch16_224(pretrained=True)
    base_model = model

    print("TUNING METHOD: ", args.tuning_method)

    # Create the mask

    if args.tuning_method == "tune_attention_blocks_random":
        mask_length = len(model.blocks)
    elif args.tuning_method == "tune_attention_params_random":
        mask_length = len(model.blocks) * 4

    print("Creating mask of length: ", mask_length)
    args.mask = utils.create_random_mask(
        mask_length, args.mask_gen_method, device, sigma=args.sigma
    )
    initial_mask = args.mask

    print("Initial Mask: ", initial_mask)
    
    keys = ["mask_el_" + str(i) for i in range(mask_length)]
    values = [[] for i in range(mask_length)]
    MASK_DICT = {
        key: value for key, value in zip(keys, values)
    }  # A dictionary to store the values of each mask param during training
    BINARY_MASK_DICT = {
        key: value for key, value in zip(keys, values)
    }  # A dictionary to store the values of each binary mask element during training
    BINARY_MASK_PLOT_ARRAYS = []

    ALL_THRESHOLDS = []

    # Track the original mask and binary mask
    MASK_DICT = track_mask(args.mask, MASK_DICT)

    if args.use_adaptive_threshold:
        _thr = np.mean(args.mask.detach().cpu().numpy())
        threshold = _thr
        print("Threshold: ", threshold)
        ALL_THRESHOLDS.append(threshold)

        if args.wandb_logging:
            wandb.log({"Threshold": threshold})
    else:
        threshold = 1.0

    if args.use_gumbel_sigmoid:
        binary_mask = gumbel_sigmoid(args.mask, hard=True)
    else:
        binary_mask = args.mask >= threshold
    print("Initial Binary Mask: ", binary_mask)

    binary_mask = binary_mask.long()

    BINARY_MASK_DICT = track_mask(binary_mask, BINARY_MASK_DICT)
    binary_mask_fig_arr = plot_binary_mask(binary_mask)
    BINARY_MASK_PLOT_ARRAYS.append(binary_mask_fig_arr)

    # Disabling all parameters except attention
    if("attention" in args.tuning_method):
        for name, param in model.named_parameters():
            if "attn" not in name:
                param.requires_grad = False
    else:
        raise NotImplementedError

    enable_module(model.head)

    trainable_params, all_param = utils.check_tunable_params(model, True)
    trainable_percentage = 100 * trainable_params / all_param

    # Track the original trainable percentage
    if args.wandb_logging:
        wandb.log({"Trainable Percentage": trainable_percentage})

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="none")
    ece_criterion = utils.ECELoss()

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in [
            "class_token",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay
        if len(custom_keys_weight_decay) > 0
        else None,
    )

    # Optimizer
    inner_optimizer = get_optimizer(args, parameters)
    outer_optimizer = get_optimizer(args, [args.mask], meta=True)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # LR Scheduler
    lr_scheduler_inner = get_lr_scheduler(args, inner_optimizer)

    if not args.lr_scheduler_outer == "constant":
        lr_scheduler_outer = get_lr_scheduler(args, outer_optimizer)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # From training_utils.py
    model_ema = get_model_ema(model_without_ddp, args)
    print("model ema", model_ema)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            inner_optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler_inner.load_state_dict(checkpoint["lr_scheduler"])
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
            if(args.sens_attribute == 'gender'):
                test_acc, test_male_acc, test_female_acc, test_loss, test_max_loss = evaluate_fairness_gender(
                    model_ema,
                    criterion,
                    ece_criterion,
                    data_loader_test,
                    args=args,
                    device=device,
                    log_suffix="EMA",
                    pos_label=0,
                )
            elif(args.sens_attribute == 'skin_type'):
                (
                test_acc,
                test_acc_type0,
                test_acc_type1,
                test_acc_type2,
                test_acc_type3,
                test_acc_type4,
                test_acc_type5,
                test_loss,
                test_max_loss,
            ) = evaluate_fairness_skin_type(
                    model_ema,
                    criterion,
                    ece_criterion,
                    data_loader_test,
                    args=args,
                    device=device,
                    log_suffix="EMA",
                    pos_label=0,
            )
            elif(args.sens_attribute == 'age'):
                (
                test_acc,
                test_acc_type0,
                test_acc_type1,
                test_acc_type2,
                test_acc_type3,
                test_acc_type4,
                test_loss,
                test_max_loss,
                ) = evaluate_fairness_age(
                        model_ema,
                        criterion,
                        ece_criterion,
                        data_loader_test,
                        args=args,
                        device=device,
                        log_suffix="EMA",
                        pos_label=0,
                )
            else:
                raise NotImplementedError
        else:
            if(args.sens_attribute == 'gender'):
                test_acc, test_male_acc, test_female_acc, test_loss, test_max_loss = evaluate_fairness_gender(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_test,
                    args=args,
                    device=device,
                    pos_label=0,
                )
            elif(args.sens_attribute == 'skin_type'):
                (
                    test_acc,
                    test_acc_type0,
                    test_acc_type1,
                    test_acc_type2,
                    test_acc_type3,
                    test_acc_type4,
                    test_acc_type5,
                    test_loss,
                    test_max_loss,
                ) = evaluate_fairness_skin_type(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_test,
                    args=args,
                    device=device,
                    pos_label=0,
                )
            elif(args.sens_attribute == 'age'):
                (
                    test_acc,
                    test_acc_type0,
                    test_acc_type1,
                    test_acc_type2,
                    test_acc_type3,
                    test_acc_type4,
                    test_loss,
                    test_max_loss,
                ) = evaluate_fairness_age(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_test,
                    args=args,
                    device=device,
                    pos_label=0,
                )
        return

    # INNER LOOP: TRAINING PROCESS HERE

    if args.disable_training:
        print("Training Process Skipped")
    else:
        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.wandb_logging:
                wandb.log({"Epoch": epoch})

            loaders = zip(data_loader, cycle(data_loader_val))
            print("Epoch: ", epoch)
            print("Total Epochs: ", args.epochs)
            if args.distributed:
                train_sampler.set_epoch(epoch)
            # meta_train_one_epoch(model, criterion, ece_criterion, inner_optimizer, data_loader, device, epoch, args, model_ema, scaler)

            model.train()
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter(
                "lr", utils.SmoothedValue(window_size=1, fmt="{value}")
            )
            metric_logger.add_meter(
                "img/s", utils.SmoothedValue(window_size=10, fmt="{value}")
            )

            header = f"Epoch: [{epoch}]"

            # META TRAINING: INNER LOOP
            
            if(args.sens_attribute == 'gender'):
                total_loss_male = 0.0
                total_loss_female = 0.0
                num_male = 0
                num_female = 0
            elif(args.sens_attribute == 'skin_type'):
                total_loss_type1 = 0.0
                total_loss_type2 = 0.0
                total_loss_type3 = 0.0
                total_loss_type4 = 0.0
                total_loss_type5 = 0.0
                total_loss_type6 = 0.0
                num_type1 = 0
                num_type2 = 0
                num_type3 = 0
                num_type4 = 0
                num_type5 = 0
                num_type6 = 0
            elif(args.sens_attribute == 'age'):
                total_loss_type1 = 0.0
                total_loss_type2 = 0.0
                total_loss_type3 = 0.0
                total_loss_type4 = 0.0
                total_loss_type5 = 0.0
                num_type1 = 0
                num_type2 = 0
                num_type3 = 0
                num_type4 = 0
                num_type5 = 0
            else:
                raise NotImplementedError

            for i, ((image, target, sens_attr), (image_val, target_val, sens_attr_val)) in enumerate(loaders):
                start_time = time.time()
                image, target = image.to(device), target.to(device)
                image_val, target_val = image_val.to(device), target_val.to(device)
                #sens_attr, sens_attr_val = sens_attr.to(device), sens_attr_val.to(device)

                # print(target_val)

                # import pdb
                # pdb.set_trace()

                # Reset the fast weights
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
                    # print("Attention parameters", attn_params)
                    attn_params = get_attn_params(model)
                    # print("All params trainable" if not check_trainability(attn_params) else "Not all params trainable")
                    print("Loss: ", torch.mean(loss).item())

                    # 2. Calculate the gradients (manually)
                    try:
                        grad = torch.autograd.grad(torch.mean(loss), attn_params, create_graph=True)
                    except:
                        import pdb

                        pdb.set_trace()
                    inner_lr = lr_scheduler_inner.get_last_lr()[-1]

                    if args.wandb_logging:
                        wandb.log({"Inner LR": inner_lr})

                        if not args.lr_scheduler_outer == "constant":
                            wandb.log(
                                {"Outer LR": lr_scheduler_outer.get_last_lr()[-1]}
                            )
                        else:
                            wandb.log({"Outer LR": args.outer_lr})

                    # 3. Update the attention parameters using the update equation
                    for k, weight in enumerate(attn_params):
                        if weight.fast is None:
                            if args.use_gumbel_sigmoid:
                                weight.fast = (
                                    weight
                                    - inner_lr
                                    * args.lr_scaler
                                    * gumbel_sigmoid(args.mask[k // 4], hard=True)
                                    * grad[k]
                                )
                            else:
                                weight.fast = (
                                    weight
                                    - inner_lr
                                    * args.lr_scaler
                                    * args.mask[k // 4]
                                    * grad[k]
                                )
                        else:
                            if args.use_gumbel_sigmoid:
                                weight.fast = (
                                    weight.fast
                                    - inner_lr
                                    * args.lr_scaler
                                    * gumbel_sigmoid(args.mask[k // 4], hard=True)
                                    * grad[k]
                                )
                            else:
                                weight.fast = (
                                    weight.fast
                                    - inner_lr
                                    * args.lr_scaler
                                    * args.mask[k // 4]
                                    * grad[k]
                                )

                    # 4. TODO: We might need to clip the mask between 0 and 1

                    # OUTER LOOP: META TRAINING (THIS SHOULD BE ON VALIDATION DATA) - TRAINING THE META PARAMS (MASK)
                    output = model(image_val)
                    val_loss = criterion(output, target_val)

                    if(args.sens_attribute == 'gender'):
                        # Separate losses for male and female populations
                        # loss_male = torch.mean(val_loss[sens_attr_val == 'M'])
                        # loss_female = torch.mean(val_loss[sens_attr_val == 'F'])

                        indexes_males = [index for index, gender in enumerate(sens_attr_val) if gender == 'M']
                        indexes_females = [index for index, gender in enumerate(sens_attr_val) if gender == 'F']

                        loss_male = [val_loss[index] for index in indexes_males]
                        loss_female = [val_loss[index] for index in indexes_females]
                
                        # total_loss_male += loss_male.item()
                        # total_loss_female += loss_female.item()

                        total_loss_male = sum(loss_male)
                        total_loss_female = sum(loss_female)
                    
                        #num_male += torch.sum(sens_attr_val == 'M').item()
                        #num_female += torch.sum(sens_attr_val == 'F').item()
                        num_male += sens_attr_val.count('M')
                        num_female += sens_attr_val.count('F')
                            
                        avg_loss_male = total_loss_male / num_male if num_male > 0 else 0.0
                        avg_loss_female = total_loss_female / num_female if num_female > 0 else 0.0

                        # Take the maximum of the two losses
                        meta_loss = max(avg_loss_male, avg_loss_female)

                    elif(args.sens_attribute == 'skin_type'):
                        # Separate losses for each skin type

                        sens_attr_val = sens_attr_val.tolist()

                        idx_type1 = [index for index, skin_type in enumerate(sens_attr_val) if skin_type == 0]
                        idx_type2 = [index for index, skin_type in enumerate(sens_attr_val) if skin_type == 1]
                        idx_type3 = [index for index, skin_type in enumerate(sens_attr_val) if skin_type == 2]
                        idx_type4 = [index for index, skin_type in enumerate(sens_attr_val) if skin_type == 3]
                        idx_type5 = [index for index, skin_type in enumerate(sens_attr_val) if skin_type == 4]
                        idx_type6 = [index for index, skin_type in enumerate(sens_attr_val) if skin_type == 5]

                        loss_type1 = [val_loss[index] for index in idx_type1]
                        loss_type2 = [val_loss[index] for index in idx_type2]
                        loss_type3 = [val_loss[index] for index in idx_type3]
                        loss_type4 = [val_loss[index] for index in idx_type4]
                        loss_type5 = [val_loss[index] for index in idx_type5]
                        loss_type6 = [val_loss[index] for index in idx_type6]

                        # loss_type1 = torch.mean(loss[sens_attr_val == 0])
                        # loss_type2 = torch.mean(loss[sens_attr_val == 1])
                        # loss_type3 = torch.mean(loss[sens_attr_val == 2])
                        # loss_type4 = torch.mean(loss[sens_attr_val == 3])
                        # loss_type5 = torch.mean(loss[sens_attr_val == 4])
                        # loss_type6 = torch.mean(loss[sens_attr_val == 5])
                
                        # total_loss_type1 += loss_type1.item()
                        # total_loss_type2 += loss_type2.item()
                        # total_loss_type3 += loss_type3.item()
                        # total_loss_type4 += loss_type4.item()
                        # total_loss_type5 += loss_type5.item()
                        # total_loss_type6 += loss_type6.item()

                        total_loss_type1 = sum(loss_type1)
                        total_loss_type2 = sum(loss_type2)
                        total_loss_type3 = sum(loss_type3)
                        total_loss_type4 = sum(loss_type4)
                        total_loss_type5 = sum(loss_type5)
                        total_loss_type6 = sum(loss_type6)

                        #num_type1 += torch.sum(sens_attr_val == 0).item()
                        num_type1 += sens_attr_val.count(0)
                        num_type2 += sens_attr_val.count(1)
                        num_type3 += sens_attr_val.count(2)
                        num_type4 += sens_attr_val.count(3)
                        num_type5 += sens_attr_val.count(4)
                        num_type6 += sens_attr_val.count(5)

                        # num_type2 += torch.sum(sens_attr_val == 1).item()
                        # num_type3 += torch.sum(sens_attr_val == 2).item()
                        # num_type4 += torch.sum(sens_attr_val == 3).item()
                        # num_type5 += torch.sum(sens_attr_val == 4).item()
                        # num_type6 += torch.sum(sens_attr_val == 5).item()

                        avg_loss_type1 = total_loss_type1 / num_type1 if num_type1 > 0 else 0.0
                        avg_loss_type2 = total_loss_type2 / num_type2 if num_type2 > 0 else 0.0
                        avg_loss_type3 = total_loss_type3 / num_type3 if num_type3 > 0 else 0.0
                        avg_loss_type4 = total_loss_type4 / num_type4 if num_type4 > 0 else 0.0
                        avg_loss_type5 = total_loss_type5 / num_type5 if num_type5 > 0 else 0.0
                        avg_loss_type6 = total_loss_type6 / num_type6 if num_type6 > 0 else 0.0

                        # Take the maximum of all the losses
                        meta_loss = max(avg_loss_type1, avg_loss_type2, avg_loss_type3, avg_loss_type4, avg_loss_type5, avg_loss_type6)

                    elif(args.sens_attribute == 'age'):
                        # Separate losses for each age group

                        # Separate losses for each age group

                        sens_attr_val = sens_attr_val.tolist()

                        idx_type1 = [index for index, _age_group in enumerate(sens_attr_val) if _age_group == 0]
                        idx_type2 = [index for index, _age_group in enumerate(sens_attr_val) if _age_group == 1]
                        idx_type3 = [index for index, _age_group in enumerate(sens_attr_val) if _age_group == 2]
                        idx_type4 = [index for index, _age_group in enumerate(sens_attr_val) if _age_group == 3]
                        idx_type5 = [index for index, _age_group in enumerate(sens_attr_val) if _age_group == 4]

                        loss_type1 = [val_loss[index] for index in idx_type1]
                        loss_type2 = [val_loss[index] for index in idx_type2]
                        loss_type3 = [val_loss[index] for index in idx_type3]
                        loss_type4 = [val_loss[index] for index in idx_type4]
                        loss_type5 = [val_loss[index] for index in idx_type5]

                        total_loss_type1 = sum(loss_type1)
                        total_loss_type2 = sum(loss_type2)
                        total_loss_type3 = sum(loss_type3)
                        total_loss_type4 = sum(loss_type4)
                        total_loss_type5 = sum(loss_type5)

                        #num_type1 += torch.sum(sens_attr_val == 0).item()
                        num_type1 += sens_attr_val.count(0)
                        num_type2 += sens_attr_val.count(1)
                        num_type3 += sens_attr_val.count(2)
                        num_type4 += sens_attr_val.count(3)
                        num_type5 += sens_attr_val.count(4)

                        avg_loss_type1 = total_loss_type1 / num_type1 if num_type1 > 0 else 0.0
                        avg_loss_type2 = total_loss_type2 / num_type2 if num_type2 > 0 else 0.0
                        avg_loss_type3 = total_loss_type3 / num_type3 if num_type3 > 0 else 0.0
                        avg_loss_type4 = total_loss_type4 / num_type4 if num_type4 > 0 else 0.0
                        avg_loss_type5 = total_loss_type5 / num_type5 if num_type5 > 0 else 0.0

                        # Take the maximum of all the losses
                        meta_loss = max(avg_loss_type1, avg_loss_type2, avg_loss_type3, avg_loss_type4, avg_loss_type5)
                    else:
                        raise NotImplementedError

                    
                    print("META LOSS: ", meta_loss.item())
                    
                    if args.wandb_logging:
                        wandb.log({"Meta Loss": meta_loss.item()})

                    outer_optimizer.zero_grad()
                    meta_loss.backward(retain_graph=True)
                    outer_optimizer.step()

                    # Reset the fast weights
                    for k, weight in enumerate(attn_params):
                        weight.fast = None

                    # STANDARD UPDATE: Training the inner loop again for better training

                    # TODO: THRESHOLD THE MASK HERE
                    # TODO: CAN USE DIFFERENT SCHEMES: 1. SIGMOID

                    if args.use_adaptive_threshold:
                        _thr = np.mean(args.mask.detach().cpu().numpy())
                        threshold = (
                            args.thr_ema_decay * _thr
                            + (1 - args.thr_ema_decay) * ALL_THRESHOLDS[-1]
                        )
                        print("Threshold: ", threshold)

                        if args.wandb_logging:
                            wandb.log({"Threshold": threshold})
                    else:
                        threshold = 1.0

                    # THRESHOLD THE MASK

                    # if(args.apply_sigmoid_to_mask):
                    #     # Apply sigmoid to the mask
                    #     args.mask = torch.sigmoid(args.mask)

                    if args.use_gumbel_sigmoid:
                        binary_mask = gumbel_sigmoid(args.mask, hard=True)
                    else:
                        binary_mask = args.mask >= threshold
                        binary_mask = binary_mask.long()

                    # if args.wandb_logging:
                    #     binary_mask_fig_arr = plot_binary_mask(binary_mask)
                    #     BINARY_MASK_PLOT_ARRAYS.append(binary_mask_fig_arr)

                    ## APPLY THE UPDATED MASK
                    # TODO: Implement different methods for applying mask here
                    if args.tuning_method == "tune_attention_blocks_random":
                        for idx, block in enumerate(model.blocks):
                            if binary_mask[idx] == 1:
                                enable_module(block.attn)
                            else:
                                disable_module(block.attn)
                    elif args.tuning_method == "tune_attention_params_random":
                        attn_params = [
                            p
                            for name_p, p in model.named_parameters()
                            if ".attn." in name_p or "attention" in name_p
                        ]
                        for idx, p in enumerate(attn_params):
                            if binary_mask[idx] == 1:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False

                    trainable_params, all_param = check_tunable_params(model, False)
                    trainable_percentage = 100 * trainable_params / all_param
                    track_trainable_params.append(trainable_percentage)

                        # x = np.arange(len(temp_mask))
                        # for i, value in enumerate(temp_mask):
                        #     wandb.log({f"Mask Param {str(i)}": wandb.plot.line(x=x, y=[value], labels=[f"Mask Param {str(i)}"])})

                    # STANDARD UPDATE
                    print("STANDARD UPDATE")
                    output = model(image)
                    loss = criterion(output, target)
                    inner_optimizer.zero_grad()
                    torch.mean(loss).backward()
                    inner_optimizer.step()

                    acc1, acc5 = utils.accuracy(
                        output, target, topk=(1, args.num_classes)
                    )
                    # auc = utils.auc(output, target)
                    print("ACC1: {}, ACC5: {}, LOSS: {}".format(acc1, acc5, torch.mean(loss)))
                    if(args.sens_attribute == 'gender'):
                        val_acc, val_male_acc, val_female_acc, val_loss, val_max_loss = evaluate_fairness_gender(
                            model,
                            criterion,
                            ece_criterion,
                            data_loader_val,
                            args=args,
                            device=device,
                        )
                        print(
                            "Val Acc: {:.2f}, Val Male Acc {:.2f}, Val Female Acc {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                                val_acc, val_male_acc, val_female_acc, torch.mean(val_loss), val_max_loss
                            )
                        )
                    elif(args.sens_attribute == 'skin_type'):
                        (
                            val_acc,
                            val_acc_type0,
                            val_acc_type1,
                            val_acc_type2,
                            val_acc_type3,
                            val_acc_type4,
                            val_acc_type5,
                            val_loss,
                            val_max_loss,
                        ) = evaluate_fairness_skin_type(
                            model,
                            criterion,
                            ece_criterion,
                            data_loader_val,
                            args=args,
                            device=device,
                        )
                        print(
                                "Val Acc: {:.2f}, Val Type 0 Acc: {:.2f}, Val Type 1 Acc: {:.2f}, Val Type 2 Acc: {:.2f}, Val Type 3 Acc: {:.2f}, Val Type 4 Acc: {:.2f}, Val Type 5 Acc: {:.2f}, Val MAX LOSS: {:.2f}".format(
                                    val_acc,
                                    val_acc_type0,
                                    val_acc_type1,
                                    val_acc_type2,
                                    val_acc_type3,
                                    val_acc_type4,
                                    val_acc_type5,
                                    torch.mean(val_loss),
                                    val_max_loss,
                                )
                            )
                    elif(args.sens_attribute == 'age'):
                        (
                            val_acc,
                            val_acc_age0,
                            val_acc_age1,
                            val_acc_age2,
                            val_acc_age3,
                            val_acc_age4,
                            val_loss,
                            val_max_loss,
                        ) = evaluate_fairness_age(
                            model,
                            criterion,
                            ece_criterion,
                            data_loader_val,
                            args=args,
                            device=device,
                        )
                        print(
                                "Val Acc: {:.2f}, Val Type 0 Acc: {:.2f}, Val Type 1 Acc: {:.2f}, Val Type 2 Acc: {:.2f}, Val Type 3 Acc: {:.2f}, Val Type 4 Acc: {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                                    val_acc,
                                    val_acc_age0,
                                    val_acc_age1,
                                    val_acc_age2,
                                    val_acc_age3,
                                    val_acc_age4,
                                    torch.mean(val_loss),
                                    val_max_loss,
                                )
                            )
                    else:
                        raise NotImplementedError

                    if args.wandb_logging:
                        wandb.log({"Train Accuracy": acc1})
                        wandb.log({"Standard Update Loss": loss})
                        wandb.log({"Val Accuracy": val_acc})

                        if(args.sens_attribute == 'gender'):
                            wandb.log({"Val Male Accuracy": val_male_acc})
                            wandb.log({"Val Female Accuracy": val_female_acc})
                            wandb.log({"Val Loss": torch.mean(val_loss)})
                        elif(args.sens_attribute == 'skin_type'):
                            pass
                        elif(args.sens_attribute == 'age'):
                            pass
                        #wandb.log({"Val AUC": val_auc})

                    # Re-enabling all the attention blocks
                    for idx, block in enumerate(model.blocks):
                        enable_module(block.attn)

                    # if args.use_gumbel_sigmoid:
                    #     print("MASK: ", gumbel_sigmoid(args.mask))
                    # else:
                    #     print("MASK: ", args.mask)
                    
                if(args.sens_attribute == 'gender'):
                    acc1, acc_male, acc_female = utils.accuracy_by_gender(output, target, sens_attr, topk=(1, args.num_classes))
                    acc1 = acc1[0]
                    acc_male = acc_male[0]
                    acc_female = acc_female[0]

                    batch_size = image.shape[0]
                    metric_logger.meters["acc1_male"].update(acc_male.item(), n=batch_size)
                    metric_logger.meters["acc1_female"].update(acc_female.item(), n=batch_size)
                elif(args.sens_attribute == 'skin_type'):
                    acc1, res_type0, res_type1, res_type2, res_type3, res_type4, res_type5 = utils.accuracy_by_skin_type(
                            output, target, sens_attr, topk=(1,), num_skin_types=args.num_skin_types
                        )
                    acc1 = acc1[0]
                    acc_type0 = res_type0[0]
                    acc_type1 = res_type1[0]
                    acc_type2 = res_type2[0]
                    acc_type3 = res_type3[0]
                    acc_type4 = res_type4[0]
                    acc_type5 = res_type5[0]

                    batch_size = image.shape[0]
                    metric_logger.meters["acc_type0"].update(acc_type0.item(), n=batch_size)
                    metric_logger.meters["acc_type1"].update(acc_type1.item(), n=batch_size)
                    metric_logger.meters["acc_type2"].update(acc_type2.item(), n=batch_size)
                    metric_logger.meters["acc_type3"].update(acc_type3.item(), n=batch_size)
                    metric_logger.meters["acc_type4"].update(acc_type4.item(), n=batch_size)
                    metric_logger.meters["acc_type5"].update(acc_type5.item(), n=batch_size)
                elif(args.sens_attribute == 'age'):
                    acc1, res_type0, res_type1, res_type2, res_type3, res_type4 = utils.accuracy_by_age(
                            output, target, sens_attr, topk=(1,), 
                        )
                    acc1 = acc1[0]
                    acc_type0 = res_type0[0]
                    acc_type1 = res_type1[0]
                    acc_type2 = res_type2[0]
                    acc_type3 = res_type3[0]
                    acc_type4 = res_type4[0]

                    batch_size = image.shape[0]
                    metric_logger.meters["acc_Age0"].update(acc_type0.item(), n=batch_size)
                    metric_logger.meters["acc_Age1"].update(acc_type1.item(), n=batch_size)
                    metric_logger.meters["acc_Age2"].update(acc_type2.item(), n=batch_size)
                    metric_logger.meters["acc_Age3"].update(acc_type3.item(), n=batch_size)
                    metric_logger.meters["acc_Age4"].update(acc_type4.item(), n=batch_size)
                else:
                    raise NotImplementedError
                
                
                metric_logger.update(
                    loss=torch.mean(loss).item(), lr=inner_optimizer.param_groups[0]["lr"]
                )
                metric_logger.update(
                    ece_loss=ece_loss.item(), lr=inner_optimizer.param_groups[0]["lr"]
                )
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

                metric_logger.meters["img/s"].update(
                    batch_size / (time.time() - start_time)
                )

            ############################ Logging at epoch level ############################
            
            print("\n")
            print("############################ EPOCH FINISHED ############################")

            print("Mean Trainable Percentage: ", np.mean(track_trainable_params))

            if args.wandb_logging:
                wandb.log({"Mean Trainable Percentage": np.mean(track_trainable_params)})

            # if args.wandb_logging:
            #     # Separately log every element of the mask vector to wandb
            #     temp_mask = args.mask.detach().cpu().numpy()

            #     for i in range(len(args.mask)):
            #         wandb.log(
            #             {"Mask Parameter {}".format(str(i)): temp_mask[i]}
            #         )

            if args.wandb_logging:
                
                _df_binary_mask = pd.DataFrame(BINARY_MASK_DICT)
                binary_table = wandb.Table(dataframe=_df_binary_mask)
                wandb.log({"Mask Params": binary_table})

            if args.use_gumbel_sigmoid:
                print("MASK: ", gumbel_sigmoid(args.mask))
                print("Binary Mask", gumbel_sigmoid(args.mask, hard=True))
            else:
                print("MASK: ", args.mask)

            print("\n")

            MASK_DICT = track_mask(args.mask, MASK_DICT)
            BINARY_MASK_DICT = track_mask(binary_mask, BINARY_MASK_DICT)

            lr_scheduler_inner.step()

            if not args.lr_scheduler_outer == "constant":
                lr_scheduler_outer.step()

            if model_ema:
                if(args.sens_attribute == 'gender'):
                    val_acc, val_male_acc, val_female_acc, val_loss, val_max_loss = evaluate_fairness_gender(
                        model,
                        criterion,
                        ece_criterion,
                        data_loader_val,
                        args=args,
                        device=device,
                    )
                else:
                    raise NotImplementedError
            if args.output_dir:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": inner_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler_inner.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                if model_ema:
                    checkpoint["model_ema"] = model_ema.state_dict()
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                # utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoints', f"model_{epoch}.pth"))7
                ckpt_path = os.path.join(
                    args.output_dir,
                    "checkpoints",
                    "meta_checkpoint_" + args.tuning_method + ".pth",
                )

                if not args.disable_checkpointing:
                    utils.save_on_master(checkpoint, ckpt_path)

        print("Training Finished")

        # if(args.wandb_logging):
        #     # Uploading Binary Mask Plot
        #     BINARY_MASK_PLOT_ARRAYS = np.vstack(BINARY_MASK_PLOT_ARRAYS)
        #     plot_img = Image.fromarray(BINARY_MASK_PLOT_ARRAYS)
        #     wandb.log({"Binary Mask Plot": wandb.Image(plot_img)})

        #     # Uploading Line Plot for mask during training
        #     _df_mask = pd.DataFrame(MASK_DICT)
        #     cols = list(_df_mask.columns)
        #     for i in range(len(cols)):
        #         plt.plot(_df_mask[cols[i]], label=cols[i])

        #     plt.legend()
        #     plt.xlabel("Training Steps")
        #     plt.ylabel("Mask Value")
        #     plt.title("Change in Mask Values during Training")
        #     # plot_path = 'Mask_plot_' + args.dataset + '_MaskGen_' + args.mask_gen_method + '.png'
        #     plot_path = os.path.join(args.fig_savepath, 'Training_mask_plot.png')
        #     plt.savefig(plot_path)

        #     wandb.log({"Training Mask Plot": wandb.Image(plot_path)})

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        # Plotting the change in mask during training
        plot_mask(args, MASK_DICT)
        
        if(args.sens_attribute == 'gender'):
            val_acc, val_male_acc, val_female_acc, val_loss, val_max_loss = evaluate_fairness_gender(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )
            print("Val accuracy: ", val_acc)
            print("Val Male acc: ", val_male_acc)
            print("Val Female acc: ", val_female_acc)
            print("Val loss: ", torch.mean(val_loss))
            print("Val max loss: ", val_max_loss)

        elif(args.sens_attribute == 'skin_type'):
            (
                val_acc,
                val_acc_type0,
                val_acc_type1,
                val_acc_type2,
                val_acc_type3,
                val_acc_type4,
                val_acc_type5,
                val_loss,
                val_max_loss,
            ) = evaluate_fairness_skin_type(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

            print("Val accuracy: ", val_acc)
            print("Val Type 0 acc: ", val_acc_type0)
            print("Val Type 1 acc: ", val_acc_type1)
            print("Val Type 2 acc: ", val_acc_type2)
            print("Val Type 3 acc: ", val_acc_type3)
            print("Val Type 4 acc: ", val_acc_type4)
            print("Val Type 5 acc: ", val_acc_type5)
            print("Val loss: ", torch.mean(val_loss))
            print("Val max loss: ", val_max_loss)
        elif(args.sens_attribute == 'age'):
            (
                val_acc,
                val_acc_type0,
                val_acc_type1,
                val_acc_type2,
                val_acc_type3,
                val_acc_type4,
                val_loss,
                val_max_loss,
            ) = evaluate_fairness_age(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

            print("Val accuracy: ", val_acc)
            print("Val Age Group 0 acc: ", val_acc_type0)
            print("Val Age Group 1 acc: ", val_acc_type1)
            print("Val Age Group 2 acc: ", val_acc_type2)
            print("Val Age Group 3 acc: ", val_acc_type3)
            print("Val Age Group 4 acc: ", val_acc_type4)
            print("Val loss: ", torch.mean(val_loss))
            print("Val max loss: ", val_max_loss)
        else:
            raise NotImplementedError


        if(args.sens_attribute == 'gender'):
            test_acc, test_male_acc, test_female_acc, test_loss, test_max_loss = evaluate_fairness_gender(
                model,
                criterion,
                ece_criterion,
                data_loader_test,
                args=args,
                device=device,
                pos_label=0,
            )
            print("\n")
            print("Test accuracy: ", test_acc)
            print("Test Male Accuracy: ", test_male_acc)
            print("Test Female Accuracy: ", test_female_acc)

        elif(args.sens_attribute == 'skin_type'):
            (
                test_acc,
                test_acc_type0,
                test_acc_type1,
                test_acc_type2,
                test_acc_type3,
                test_acc_type4,
                test_acc_type5,
                test_loss,
                test_max_loss,
            ) = evaluate_fairness_skin_type(
                model,
                criterion,
                ece_criterion,
                data_loader_test,
                args=args,
                device=device,
            )
            print("\n")
            print("Test Type 0 Accuracy: ", test_acc_type0)
            print("Test Type 1 Accuracy: ", test_acc_type1)
            print("Test Type 2 Accuracy: ", test_acc_type2)
            print("Test Type 3 Accuracy: ", test_acc_type3)
            print("Test Type 4 Accuracy: ", test_acc_type4)
            print("Test Type 5 Accuracy: ", test_acc_type5)

        elif(args.sens_attribute == 'age'):
            (
                test_acc,
                test_acc_type0,
                test_acc_type1,
                test_acc_type2,
                test_acc_type3,
                test_acc_type4,
                test_loss,
                test_max_loss,
            ) = evaluate_fairness_age(
                model,
                criterion,
                ece_criterion,
                data_loader_test,
                args=args,
                device=device,
            )
            print("\n")
            print("Test Age Group 0 Accuracy: ", test_acc_type0)
            print("Test Age Group 1 Accuracy: ", test_acc_type1)
            print("Test Age Group 2 Accuracy: ", test_acc_type2)
            print("Test Age Group 3 Accuracy: ", test_acc_type3)
            print("Test Age Group 4 Accuracy: ", test_acc_type4)
        else:
            raise NotImplementedError

        print("Test accuracy: ", test_acc)
        print("Test loss: ", round(torch.mean(test_loss).item(), 3))
        print("Test max loss: ", round(test_max_loss.item(), 3))

        # print("Initial Mask: ", initial_mask)
        # print("Final Mask: ", args.mask)
        # print("Difference Mask: ", initial_mask.detach().cpu().numpy() - args.mask.detach().cpu().numpy())

        # STD for each element in the mask during training
        mask_el_df = pd.DataFrame(columns=["Element", "Standard Deviation"])
        for key in MASK_DICT.keys():
            print("STD for ", key, ": ", np.std(MASK_DICT[key]))
            mask_el_df.loc[len(mask_el_df)] = [key, np.std(MASK_DICT[key])]

            if args.wandb_logging:
                mask_el_table = wandb.Table(
                    dataframe=mask_el_df, allow_mixed_types=True
                )
                wandb.log({"Hyper Parameters": mask_el_table})

        mask_el_df2 = pd.DataFrame().from_dict(MASK_DICT)
        print(
            "Saving Mask Tracking Dataframe at: ",
            os.path.join(
                args.fig_savepath,
                "Mask_Elements_"
                + args.tuning_method
                + "_"
                + args.model
                + str(args.outer_lr)
                + ".csv",
            ),
        )
        mask_el_df2.to_csv(
            os.path.join(
                args.fig_savepath,
                "Mask_Elements_"
                + args.tuning_method
                + "_"
                + args.model
                + "_"
                + str(args.outer_lr)
                + ".csv",
            ),
            index=False,
        )

        # columns=['Tuning Method','Train Percent','LR Scaler', 'Inner LR', 'Outer LR','Test Acc@1','Vector Path']
        if args.use_gumbel_sigmoid:
            method_name = "Dynamic_Gumbel_" + args.tuning_method
        else:
            method_name = "Dynamic_" + args.tuning_method

        if(args.sens_attribute == 'gender'):
            new_row = [
                method_name,
                round(np.mean(track_trainable_params), 3),
                args.lr_scaler,
                args.lr,
                args.outer_lr,
                test_acc,
                test_male_acc,
                test_female_acc,
                round(abs(test_male_acc - test_female_acc), 3),
                np.nan
            ]
        else:
            if(args.sens_attribute == "skin_type"):
                best_acc = max(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4, test_acc_type5)
                worst_acc = min(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4, test_acc_type5)
            elif(args.sens_attribute == "age"):
                best_acc = max(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4)
                worst_acc = min(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4)

            new_row = [
                method_name,
                round(np.mean(track_trainable_params), 3),
                args.lr_scaler,
                args.lr,
                args.outer_lr,
                test_acc,
                best_acc,
                worst_acc,
                round(abs(best_acc - worst_acc), 3),
                np.nan
            ]
        
        test_results_df.loc[len(test_results_df)] = new_row
        test_results_df.to_csv(
            os.path.join(args.output_dir, args.test_results_df), index=False
        )

        print("Saving Binary Vector at: ", args.binary_mask_savepath)
        #Making the directory if it doesn't exist
        if not os.path.exists(args.binary_mask_savepath):
            os.makedirs(args.binary_mask_savepath)
        mask_name = "test_acc_{:.2f}_outerlr_{:.4f}_scaler_{}".format(test_acc, args.outer_lr, int(args.lr_scaler)) + ".npy"
        np.save(os.path.join(args.binary_mask_savepath, mask_name), gumbel_sigmoid(args.mask, hard=True).detach().cpu().numpy())



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.output_dir = os.path.join(os.getcwd(), args.model, args.dataset)
    # args.results_df = 'Fixed_Vectors_' + args.tuning_method + '_' + args.model + '_' + str(args.lr) + '.csv'
    args.results_df = "Fixed_Vectors_" + args.tuning_method + "_" + args.model + ".csv"
    args.test_results_df = (
        "Fairness_Test_set_results_" + args.sens_attribute + "_" + args.tuning_method + "_" + args.model + ".csv"
    )
    current_wd = os.getcwd()
    args.vector_savepath = os.path.join(
        current_wd,
        "saved_vectors",
        args.model,
        args.dataset,
        args.tuning_method + "_" + str(args.lr),
    )
    args.binary_mask_savepath = os.path.join(
        current_wd,
        "Saved_Binary_Vectors_Subnetwork",
        args.model,
        args.dataset,
        args.tuning_method,
    )
    args.fig_savepath = os.path.join(args.output_dir, "plots/")

    if args.masking_vector_idx is None and args.tuning_method != "fullft":
        args.save_flag = True
    else:
        args.save_flag = False

    args.val_split = 0.2

    # Saving the hparams to wandb
    hparams_df = pd.DataFrame(columns=["Hparams", "Value"])
    hparams_df.loc[len(hparams_df)] = ["Inner LR", args.lr]
    hparams_df.loc[len(hparams_df)] = ["Outer LR", args.outer_lr]
    hparams_df.loc[len(hparams_df)] = ["LR Scaler", args.lr_scaler]
    hparams_df.loc[len(hparams_df)] = ["Tuning Method", str(args.tuning_method)]
    hparams_df.loc[len(hparams_df)] = ["LR Warmup Epochs", args.lr_warmup_epochs]
    hparams_df.loc[len(hparams_df)] = ["Sigma", args.sigma]
    hparams_df.loc[len(hparams_df)] = ["Mask Gen Method", args.mask_gen_method]
    hparams_df.loc[len(hparams_df)] = [
        "Adaptive Threshold",
        args.use_adaptive_threshold,
    ]
    hparams_df.loc[len(hparams_df)] = ["Dataset", args.dataset]

    print(hparams_df)

    args.hparams_df = hparams_df

    if args.use_gumbel_sigmoid:
        args.mask_gen_method = "random_gumbel"

    main(args)
