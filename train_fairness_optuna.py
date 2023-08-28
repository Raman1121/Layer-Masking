import os

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
import torch
import torch.utils.data
import torchvision
import transforms
from utils import *
from training_utils import *
from parse_args import *
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import yaml
from pprint import pprint

import optuna
from optuna.trial import TrialState

############################################################################

def create_opt_mask(trial, args, num_blocks):

    mask_length = None
    if(args.tuning_method == 'tune_blocks' or args.tuning_method == 'tune_attention_blocks_random'):
        mask_length = num_blocks
    elif(args.tuning_method == 'tune_attention_params_random'):
        mask_length = num_blocks * 4
    else:
        raise NotImplementedError

    mask = np.zeros(mask_length, dtype=np.int8)

    for i in range(mask_length):
        mask[i] = trial.suggest_int("mask_el_{}".format(i), 0, 1)

    return mask

def create_model(args):

    print("Creating model")
    print("TUNING METHOD: ", args.tuning_method)
    model = utils.get_timm_model(args.model, num_classes=args.num_classes)

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

    return model, parameters

def create_results_df(args):

    test_results_df = None
    
    if(args.sens_attribute == 'gender'):
        test_results_df = pd.DataFrame(
                columns=[
                    "Tuning Method",
                    "Train Percent",
                    "LR Scaler",
                    "Inner LR",
                    "Test Acc@1",
                    "Test Acc Male",
                    "Test Acc Female",
                    "Test Acc Difference",
                ]
            )  
    elif(args.sens_attribute == 'skin_type' or args.sens_attribute == 'age'):
        test_results_df = pd.DataFrame(
                columns=[
                    "Tuning Method",
                    "Train Percent",
                    "LR Scaler",
                    "Inner LR",
                    "Test Acc@1",
                    "Test Acc (Best)",
                    "Test Acc (Worst)",
                    "Test Acc Difference",
                ]
            )  
    else:
        raise NotImplementedError

    return test_results_df

def define_dataloaders(args):
    with open("config.yaml") as file:
        yaml_data = yaml.safe_load(file)

    print("Creating dataset")
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

    collate_fn = None
    mixup_transforms = get_mixup_transforms(args)

    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    print("Creating data loaders")
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

    return data_loader, data_loader_val, data_loader_test

def objective(trial):

    args = get_args_parser().parse_args()
    device = torch.device(args.device)

    args.output_dir = os.path.join(os.getcwd(), args.model, args.dataset)
    args.results_df = "Fairness_Optuna_" + args.sens_attribute + "_" + args.tuning_method + "_" + args.model + ".csv"

    try:
        test_results_df = pd.read_csv(
            os.path.join(args.output_dir, args.results_df)
        )
        print("Reading existing results dataframe")
    except:
        print("Creating new results dataframe")
        test_results_df = create_results_df(args)

    args.distributed = False
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # Create datasets and dataloaders here
    data_loader, data_loader_val, data_loader_test = define_dataloaders(args)

    # Create the model here
    model, parameters = create_model(args)

    mask = create_opt_mask(trial, args, len(model.blocks))
    print("Mask: ", mask)
    print("\n")

    masking_vector = utils.get_masked_model(model, args.tuning_method, mask=list(mask))

    trainable_params, all_param = utils.check_tunable_params(model, True)
    trainable_percentage = 100 * trainable_params / all_param

    model.to(device)

    # Create the optimizer, criterion, lr_scheduler here
    criterion = nn.CrossEntropyLoss(
                label_smoothing=args.label_smoothing, reduction="none"
            )
    ece_criterion = utils.ECELoss()
    optimizer = get_optimizer(args, parameters)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None    

    lr_scheduler = get_lr_scheduler(args, optimizer)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    model_ema = None

    for epoch in range(args.start_epoch, args.epochs):
        
        train_one_epoch_fairness(
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

        if args.sens_attribute == "gender":
            (
                val_acc,
                val_male_acc,
                val_female_acc,
                val_loss,
                val_max_loss,
            ) = evaluate_fairness_gender(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )
            print(
                "Val Acc: {:.2f}, Val Male Acc {:.2f}, Val Female Acc {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                    val_acc,
                    val_male_acc,
                    val_female_acc,
                    torch.mean(val_loss),
                    val_max_loss,
                )
            )

        elif args.sens_attribute == "skin_type":
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
                "Val Acc: {:.2f}, Val Type 0 Acc: {:.2f}, Val Type 1 Acc: {:.2f}, Val Type 2 Acc: {:.2f}, Val Type 3 Acc: {:.2f}, Val Type 4 Acc: {:.2f}, Val Type 5 Acc: {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
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

        elif args.sens_attribute == "age":
            (
                val_acc,
                acc_age0_avg,
                acc_age1_avg,
                acc_age2_avg,
                acc_age3_avg,
                acc_age4_avg,
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
                "Val Acc: {:.2f}, Val Age Group0 Acc: {:.2f}, Val Age Group1 Acc: {:.2f}, Val Age Group2 Acc: {:.2f}, Val Age Group3 Acc: {:.2f}, Val Age Group4 Acc: {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                    val_acc,
                    acc_age0_avg,
                    acc_age1_avg,
                    acc_age2_avg,
                    acc_age3_avg,
                    acc_age4_avg,
                    torch.mean(val_loss),
                    val_max_loss,
                )
            )
        else:
            raise NotImplementedError

        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
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
                "checkpoint_" + args.tuning_method + ".pth",
            )
            if not args.disable_checkpointing:
                utils.save_on_master(checkpoint, ckpt_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # Obtaining the performance on val set
    print("Evaluating on the val set")

    if args.sens_attribute == "gender":
        (
            val_acc,
            val_male_acc,
            val_female_acc,
            val_loss,
            val_max_loss,
        ) = evaluate_fairness_gender(
            model,
            criterion,
            ece_criterion,
            data_loader_val,
            args=args,
            device=device,
        )

        max_acc = max(val_male_acc, val_female_acc)
        min_acc = min(val_male_acc, val_female_acc)
        acc_diff = abs(max_acc - min_acc)

        print("\n")
        print("Val Male Accuracy: ", val_male_acc)
        print("Val Female Accuracy: ", val_female_acc)
        print("Difference in sub-group performance: ", acc_diff)

    elif args.sens_attribute == "skin_type":
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

        max_acc = max(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_acc_type4)
        min_acc = min(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_acc_type4)
        acc_diff = abs(max_acc - min_acc)

        print("\n")
        print("val Type 0 Accuracy: ", val_acc_type0)
        print("val Type 1 Accuracy: ", val_acc_type1)
        print("val Type 2 Accuracy: ", val_acc_type2)
        print("val Type 3 Accuracy: ", val_acc_type3)
        print("val Type 4 Accuracy: ", val_acc_type4)
        print("val Type 5 Accuracy: ", val_acc_type5)
        print("Difference in sub-group performance: ", acc_diff)

    elif(args.sens_attribute == 'age'):
        (
            val_acc,
            acc_age0_avg,
            acc_age1_avg,
            acc_age2_avg,
            acc_age3_avg,
            acc_age4_avg,
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

        max_acc = max(acc_age0_avg, acc_age1_avg, acc_age2_avg, acc_age3_avg, acc_age4_avg)
        min_acc = min(acc_age0_avg, acc_age1_avg, acc_age2_avg, acc_age3_avg, acc_age4_avg)
        acc_diff = abs(max_acc - min_acc)

        print("\n")
        print("val Age Group 0 Accuracy: ", val_acc_type0)
        print("val Age Group 1 Accuracy: ", val_acc_type1)
        print("val Age Group 2 Accuracy: ", val_acc_type2)
        print("val Age Group 3 Accuracy: ", val_acc_type3)
        print("val Age Group 4 Accuracy: ", val_acc_type4)
        print("Difference in sub-group performance: ", acc_diff)
        
    else:
        raise NotImplementedError

    print("val overall accuracy: ", val_acc)
    print("val Max Accuracy: ", round(max_acc, 3))
    print("val Min Accuracy: ", round(min_acc, 3))
    print("val Accuracy Difference: ", round(acc_diff, 3))
    print("val loss: ", round(torch.mean(val_loss).item(), 3))
    print("val max loss: ", round(val_max_loss.item(), 3))
    
    if(args.objective_metric == 'acc_diff'):
        try:
            return acc_diff.item()
        except:
            return acc_diff
    elif(args.objective_metric == 'min_acc'):
        try:
            return min_acc.item()
        except:
            return min_acc
    elif(args.objective_metric == 'max_loss'):
        try:
            return val_max_loss.item()
        except:
            return val_max_loss

if __name__ == "__main__":
    
    args = get_args_parser().parse_args()

    if(args.objective_metric == 'acc_diff' or args.objective_metric == 'max_loss'):
        direction = 'minimize'
    elif(args.objective_metric == 'min_acc'):
        direction = 'maximize'
    else:
        raise NotImplementedError

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=3, show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Plotting
    optuna.visualization.matplotlib.plot_param_importances(study)

    optuna.visualization.matplotlib.plot_slice(study)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    

