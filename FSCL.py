import os

os.environ["TORCH_HOME"] = os.path.dirname(os.getcwd())

import datetime
import random
import re
import time
import json
import warnings
import timm
import pandas as pd
import numpy as np
import presets
import torch
import torch.utils.data
import torchvision
import transforms
from utils import *
from training_utils import *
from parse_args import *
from models import FSCL_model

from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import yaml
from pprint import pprint

def create_results_df(args):

    test_results_df = None
    
    if(args.sens_attribute == 'gender'):
        if(args.use_metric == 'acc'):
            test_results_df = pd.DataFrame(
                    columns=[
                        "Tuning Method",
                        "Train Percent",
                        "LR",
                        "Test Acc Overall",
                        "Test Acc Male",
                        "Test Acc Female",
                        "Test Acc Difference",
                        "Mask Path"
                    ]
                )  
        elif(args.use_metric == 'auc'):
            if(args.cal_equiodds):
                test_results_df = pd.DataFrame(
                        columns=[
                            "Tuning Method",
                            "Train Percent",
                            "LR",
                            "Test AUC Overall",
                            "Test AUC Male",
                            "Test AUC Female",
                            "Test AUC Difference",
                            "Mask Path",
                            "EquiOdd_diff", 
                            "EquiOdd_ratio", 
                            "DPD", 
                            "DPR", 
                        ]
                    )
            else:
                test_results_df = pd.DataFrame(
                        columns=[
                            "Tuning Method",
                            "Train Percent",
                            "LR",
                            "Test AUC Overall",
                            "Test AUC Male",
                            "Test AUC Female",
                            "Test AUC Difference",
                            "Mask Path"
                        ]
                    ) 
    elif(args.sens_attribute == 'skin_type' or args.sens_attribute == 'age' or args.sens_attribute == 'race'):
        if(args.use_metric == 'acc'):
            test_results_df = pd.DataFrame(
                    columns=[
                        "Tuning Method",
                        "Train Percent",
                        "LR",
                        "Test Acc Overall",
                        "Test Acc (Best)",
                        "Test Acc (Worst)",
                        "Test Acc Difference",
                        "Mask Path"
                    ]
                )  
        elif(args.use_metric == 'auc'):

            if(args.cal_equiodds):
                cols = ["Tuning Method", "Train Percent", "LR", "Test AUC Overall", "Test AUC (Best)", "Test AUC (Worst)", "Test AUC Difference", "EquiOdd_diff", "EquiOdd_ratio", "DPD", "DPR", "Mask Path"]
            else:
                cols = ["Tuning Method", "Train Percent", "LR", "Test AUC Overall", "Test AUC (Best)", "Test AUC (Worst)", "Test AUC Difference", "Mask Path"]
            test_results_df = pd.DataFrame(
                    columns=cols
                )  
    else:
        raise NotImplementedError

    return test_results_df

def plot_auc(args, AUC_list, best_AUC_list, worst_AUC_list, is_train):

    assert is_train is not None

    print("Saving AUC List")

    ALL_AUC = []
    ALL_BEST_AUC = []
    ALL_WORST_AUC = []

    print(AUC_list)
    print(best_AUC_list)
    print(worst_AUC_list)

    # Handling nan values, replacing them with previous values
    for i in range(len(AUC_list)):
        if(np.isnan(AUC_list[i])):
            ALL_AUC.append(AUC_list[i-1])
        else:
            ALL_AUC.append(AUC_list[i])
        
    for i in range(len(best_AUC_list)):
        if(np.isnan(best_AUC_list[i])):
            ALL_BEST_AUC.append(best_AUC_list[i-1])
        else:
            ALL_BEST_AUC.append(best_AUC_list[i])
        
    for i in range(len(worst_AUC_list)):
        if(np.isnan(worst_AUC_list[i])):
            ALL_WORST_AUC.append(worst_AUC_list[i-1])
        else:
            ALL_WORST_AUC.append(worst_AUC_list[i])

    # List for AUC Difference
    ALL_AUC_DIFF = []
    for i in range(len(ALL_AUC)):
        ALL_AUC_DIFF.append(ALL_BEST_AUC[i] - ALL_WORST_AUC[i])

    if(is_train):
        json_filename = 'AUC_' + args.dataset + '_' + args.tuning_method + '_' + args.sens_attribute + '_' + 'training.json'
    else:
        json_filename = 'AUC_' + args.dataset + '_' + args.tuning_method + '_' + args.sens_attribute + '_' + 'validation.json'

    json_dict = {
        'AUC': ALL_AUC,
        'Best AUC': ALL_BEST_AUC,
        'Worst AUC': ALL_WORST_AUC,
        'AUC Difference': ALL_AUC_DIFF
    }

    # Save the file to disk
    print("Saving json file at: ", os.path.join(args.fig_savepath, json_filename))
    with open(os.path.join(args.fig_savepath, json_filename), 'w') as fp:
        json.dump(json_dict, fp)


def plot_acc(args, ACC_list, best_ACC_list, worst_ACC_list, is_train):

    assert is_train is not None

    print("Saving ACC List")

    ALL_ACC = []
    ALL_BEST_ACC = []
    ALL_WORST_ACC = []

    print(ACC_list)
    print(best_ACC_list)
    print(worst_ACC_list)

    # Handling nan values, replacing them with previous values
    for i in range(len(ACC_list)):
        if(np.isnan(ACC_list[i])):
            ALL_ACC.append(ACC_list[i-1])
        else:
            ALL_ACC.append(ACC_list[i])
        
    for i in range(len(best_ACC_list)):
        if(np.isnan(best_ACC_list[i])):
            ALL_BEST_ACC.append(best_ACC_list[i-1])
        else:
            ALL_BEST_ACC.append(best_ACC_list[i])
        
    for i in range(len(worst_ACC_list)):
        if(np.isnan(worst_ACC_list[i])):
            ALL_WORST_ACC.append(worst_ACC_list[i-1])
        else:
            ALL_WORST_ACC.append(worst_ACC_list[i])

    # List for ACC Difference
    ALL_ACC_DIFF = []
    for i in range(len(ALL_ACC)):
        ALL_ACC_DIFF.append(ALL_BEST_ACC[i] - ALL_WORST_ACC[i])

    # Save the validation ACC, best and worst ACC as json_files
    if(is_train):
        json_filename = 'ACC_' + args.dataset + '_' + args.tuning_method + '_' + args.sens_attribute + '_' + 'training.json'
    else:
        json_filename = 'ACC_' + args.dataset + '_' + args.tuning_method + '_' + args.sens_attribute + '_' + 'validation.json'

    json_dict = {
        'ACC': ALL_ACC,
        'Best ACC': ALL_BEST_ACC,
        'Worst ACC': ALL_WORST_ACC,
        'ACC Difference': ALL_ACC_DIFF
    }

    # Save the file to disk
    print("Saving json file at: ", os.path.join(args.fig_savepath, json_filename))
    with open(os.path.join(args.fig_savepath, json_filename), 'w') as fp:
        json.dump(json_dict, fp)


def main(args):
    assert args.sens_attribute is not None, "Sensitive attribute not provided"
    
    os.makedirs(args.fig_savepath, exist_ok=True)

    # Making directory for saving checkpoints
    if args.output_dir:
        utils.mkdir(args.output_dir)
        utils.mkdir(os.path.join(args.output_dir, "checkpoints"))

    try:
        # results_df = pd.read_csv(os.path.join(args.output_dir, args.results_df))
        test_results_df = pd.read_csv(
            os.path.join(args.output_dir, args.test_results_df)
        )
        print("Reading existing results dataframe")
    except:
        print("Creating new results dataframe")
        test_results_df = create_results_df(args)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    with open("config.yaml") as file:
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

    collate_fn = None
    mixup_transforms = get_mixup_transforms(args)

    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    if(args.dataset == 'papila'):
        drop_last = False
    else:
        drop_last = True

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=drop_last
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        #drop_last=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    print("TUNING METHOD: ", args.tuning_method)
    print("Creating model")

    dim_dict = {'resnet50': 2048,
                'vit_base': 768}

    if(args.tuning_method == 'train_from_scratch'):
        #model = utils.get_timm_model(args.model, num_classes=args.num_classes, pretrained=False)
        encoder = utils.get_timm_model(args.model, num_classes=0, pretrained=False)
        model = FSCL_model.FairSupConModel(encoder=encoder, dim_in=dim_dict[args.model])
    else:
        #model = utils.get_timm_model(args.model, num_classes=args.num_classes)
        encoder = utils.get_timm_model(args.model, num_classes=0)
        model = FSCL_model.FairSupConModel(encoder=encoder, dim_in=dim_dict[args.model])

    # Calculate the sum of model parameters
    total_params = sum([p.sum() for p in model.parameters()])
    print("Sum of parameters: ", total_params)

    base_model = model

    if(args.tuning_method == 'fullft' or args.tuning_method == 'train_from_scratch'):
        pass
    elif(args.tuning_method == 'linear_readout'):
        utils.disable_module(model)
        utils.enable_module(model.head)
    else:
        assert args.mask_path is not None
        mask = np.load(args.mask_path)
        mask = get_masked_model(model, args.tuning_method, mask=mask)
        print("LOADED MASK: ", mask)

        if(np.all(np.array(mask) == 1)):  
            # If the mask contains all ones
            args.tuning_method = 'Vanilla_'+args.tuning_method
            print("Mask contains all ones. Changing tuning method to: ", args.tuning_method)

    # Check Tunable Params
    trainable_params, all_param = utils.check_tunable_params(model, True)
    trainable_percentage = 100 * trainable_params / all_param

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print("Creating FSCL Loss")
    criterion = FSCL_model.FairSupConLoss(
        temperature=args.temperature,
        contrast_mode=args.contrast_mode,
        base_temperature=args.base_temperature,
    )

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
    optimizer = get_optimizer(args, parameters)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # LR Scheduler
    lr_scheduler = get_lr_scheduler(args, optimizer)

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
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.disable_training:
        print("Training Process Skipped")

    else:

        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):

            print("EPOCH {} / {}".format(epoch, args.epochs - 1))

            if args.distributed:
                train_sampler.set_epoch(epoch)
            
            loss = train_one_epoch_fairness_FSCL(
                model,
                criterion,
                ece_criterion,
                optimizer,
                data_loader,
                device,
                epoch,
                args,
                model_ema,
                scaler,
                )
            lr_scheduler.step()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        # Save the final model
        print("Saving final model")
        torch.save(model_without_ddp.state_dict(), os.path.join(args.ckpt_dir, "FSCL_model_"+args.tuning_method+".pth"))



    if args.fscl_eval:
        # Perform evaluation on the test data using the trained classifier 
        print("Performing evaluation on the test data using the trained classifier")

        # Create criterion here
        criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing, reduction="none"
        )

        # Create model and classifier here
        # Model would remail frozen while the classifier would be trained

        model = FSCL_model.FairSupConModel(encoder=encoder, dim_in=dim_dict[args.model])

        ckpt_name = os.path.join(args.ckpt_dir, "FSCL_model_" + args.tuning_method + ".pth")
        print("Loading model from: ", ckpt_name)
        ckpt = torch.load(ckpt_name, map_location='cpu')
        
        print("Loading State Dict into the model")
        model = model.cuda()
        criterion = criterion.cuda()
        model.load_state_dict(ckpt)

        classifier = nn.Linear(dim_dict[args.model], args.num_classes)
        classifier = classifier.cuda()

        (
            train_acc,
            train_best_acc,
            train_worst_acc,
            train_auc,
            train_best_auc,
            train_worst_auc,
        ) = train_one_epoch_fairness_FSCL_classifier(
            model,
            classifier,
            criterion,
            ece_criterion,
            optimizer,
            data_loader_val,
            device,
            epoch,
            args,
            model_ema,
            scaler,
            )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.output_dir = os.path.join(os.getcwd(), args.model, args.dataset)
    args.ckpt_dir = os.path.join(args.output_dir, "FSCL_ckpt/")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if("auc" in args.objective_metric):
        args.use_metric = 'auc'
    
    if(args.use_metric == 'acc'):
        args.test_results_df = "NEW_TEST_SET_RESULTS_" + args.sens_attribute + "_" + args.objective_metric + ".csv"
    elif(args.use_metric == 'auc'):
        if(args.cal_equiodds):
            args.test_results_df = "Equiodds_AUC_NEW_TEST_SET_RESULTS_" + args.sens_attribute + "_" + args.objective_metric + ".csv"
        else:
            args.test_results_df = "AUC_NEW_TEST_SET_RESULTS_" + args.sens_attribute + "_" + args.objective_metric + ".csv"

    current_wd = os.getcwd()
    args.fig_savepath = os.path.join(args.output_dir, "plots/")

    main(args)

