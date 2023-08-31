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


def main(args):
    assert args.sens_attribute is not None, "Sensitive attribute not provided"
    if args.save_flag:
        try:
            os.makedirs(args.vector_savepath)
        except:
            print("Directory exists")

    os.makedirs(args.fig_savepath, exist_ok=True)
    masking_idx = args.masking_vector_idx
    print("Masking Index: ", masking_idx)

    """
    Different ways of reading a mask:
    1. Read from saved path (Saved vectors for random search)
    2. Read from command line
    3. Read from saved path (Saved vectors after Gradient Search)
    4. Read directly from a path (Optuna Search)
    """

    if masking_idx is not None:
        mask_filename = os.path.join(
            args.exp_vector_path, "random_vector_" + str(masking_idx) + ".npy"
        )
        mask = np.load(mask_filename)
        print(mask)
    elif args.masking_vector is not None:
        # Either read the mask from command line
        mask = read_numpy_from_cmd(args)

    elif args.subnetwork_mask_name is not None:
        # Or read the saved the mask from the path
        mask = np.load(
            os.path.join(args.binary_mask_savepath, args.subnetwork_mask_name)
        ).astype(int)
    
    elif args.mask_path is not None:
        mask = np.load(args.mask_path).astype(int)

    else:
        mask = None

    # if(args.masking_vector_idx is None):
    if args.save_flag:
        files = os.listdir(args.vector_savepath)
        files = [
            f
            for f in files
            if re.match(args.tuning_method + "_" + r"vector_\d+.npy", f)
        ]

        if len(files) > 0:
            numbers = [int(re.search(r"\d+", f).group()) for f in files]
            vector_idx = max(numbers) if numbers else None
        else:
            vector_idx = 0

        print(vector_idx, type(vector_idx))

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
        # columns=['Tuning Method','Train Percent','LR Scaler', 'Inner LR', 'Outer LR','Test Acc@1','Vector Path']
        print("Creating new results dataframe")
        if(args.sens_attribute == 'gender'):
            # results_df = pd.DataFrame(
            #     columns=[
            #         "Tuning Method",
            #         "Train Percent",
            #         "LR",
            #         "Test Acc@1",
            #         "Test Acc Male",
            #         "Test Acc Female",
            #         "Test Acc Difference",
            #     ]
            # )
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
            # results_df = pd.DataFrame(
            #     columns=[
            #         "Tuning Method",
            #         "Train Percent",
            #         "LR",
            #         "Test Acc@1",
            #         "Test Acc Male",
            #         "Test Acc Female",
            #         "Test Acc Difference",
            #     ]
            # )
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
    pprint(dataset.class_to_idx)

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

    print("Creating model")

    model = utils.get_timm_model(args.model, num_classes=args.num_classes)
    base_model = model

    print("TUNING METHOD: ", args.tuning_method)

    if args.tuning_method != "fullft":
        if args.tuning_method == "lora":
            model = utils.create_lora_model(model, block_mask=mask)

        else:
            masking_vector = utils.get_masked_model(
                model, args.tuning_method, mask=list(mask)
            )
            print("MASKING VECTOR: ", masking_vector)

    # Check Tunable Params
    trainable_params, all_param = utils.check_tunable_params(model, True)
    trainable_percentage = 100 * trainable_params / all_param

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # if(args.num_classes > 2):
    #     criterion = nn.CrossEntropyLoss(
    #         label_smoothing=args.label_smoothing, reduction="none"
    #     )
    # else:
    #     criterion = nn.BCEWithLogitsLoss(reduction="none")

    criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing, reduction="none"
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

    # if args.test_only:
    #     # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #     if args.sens_attribute == "gender":

    #         test_acc, test_male_acc, test_female_acc, test_loss, test_max_loss = evaluate_fairness_gender(
    #             model_ema,
    #             criterion,
    #             ece_criterion,
    #             data_loader_test,
    #             args=args,
    #             device=device,
    #             log_suffix="EMA",
    #             pos_label=0,
    #         )

    #     elif args.sens_attribute == "skin_type":
            
    #             test_acc, test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4, test_acc_type5, test_loss, test_max_loss = evaluate_fairness_skin_type(
    #                 model_ema,
    #                 criterion,
    #                 ece_criterion,
    #                 data_loader_test,
    #                 args=args,
    #                 device=device,
    #                 log_suffix="EMA",
    #                 pos_label=0,
    #             )
             
    #     elif args.sens_attribute == "age":
    #         raise NotImplementedError
    #     else:
    #         if args.sens_attribute == "gender":
    #             (
    #                 test_acc,
    #                 test_male_acc,
    #                 test_female_acc,
    #                 test_loss,
    #                 test_max_loss,
    #             ) = evaluate_fairness_gender(
    #                 model,
    #                 criterion,
    #                 ece_criterion,
    #                 data_loader_test,
    #                 args=args,
    #                 device=device,
    #                 pos_label=0,
    #             )
    #         elif args.sens_attribute == "skin_type":
    #             (
    #                 test_acc,
    #                 test_acc_type0,
    #                 test_acc_type1,
    #                 test_acc_type2,
    #                 test_acc_type3,
    #                 test_acc_type4,
    #                 test_acc_type5,
    #                 test_loss,
    #                 test_max_loss,
    #             ) = evaluate_fairness_skin_type(
    #                 model,
    #                 criterion,
    #                 ece_criterion,
    #                 data_loader_test,
    #                 args=args,
    #                 device=device,
    #                 pos_label=0,
    #             )
    #         elif args.sens_attribute == "age":
    #             raise NotImplementedError
    #     return

    if args.disable_training:
        print("Training Process Skipped")
    else:
        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
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
                if(args.age_type == 'multi'):
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
            elif(args.age_type == 'binary'):
                (
                    val_acc,
                    acc_age0_avg,
                    acc_age1_avg,
                    val_loss,
                    val_max_loss,
                ) = evaluate_fairness_age_binary(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_val,
                    args=args,
                    device=device,
                )
                print(
                    "Val Acc: {:.2f}, Val Age Group0 Acc: {:.2f}, Val Age Group1 Acc: {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                        val_acc,
                        acc_age0_avg,
                        acc_age1_avg,
                        torch.mean(val_loss),
                        val_max_loss,
                    )
                )
            else:
                raise NotImplementedError("Age type not supported. Choose from 'multi' or 'binary'")
            else:
                raise NotImplementedError

            # if model_ema:
            #     val_acc, val_loss, val_auc = evaluate(
            #         model_ema,
            #         criterion,
            #         ece_criterion,
            #         data_loader_val,
            #         args=args,
            #         device=device,
            #         log_suffix="EMA",
            #     )
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
        # print(f"Training time {total_time_str}")

        # Add all the information to the results_df

        # Here we are adding results on the validation set
        if args.masking_vector_idx is not None:
            if(args.sens_attribute == 'gender'):
                new_row = [
                    args.tuning_method,
                    trainable_percentage,
                    args.lr,
                    val_acc,
                    val_male_acc,
                    val_female_acc,
                    abs(val_male_acc - val_female_acc),
                    mask_filename,
                ]
            elif(args.sens_attribute == 'skin_type'):

                best_acc = max(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_acc_type4, val_acc_type5)
                worst_acc = min(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_acc_type4, val_acc_type5)

                new_row = [
                    args.tuning_method,
                    trainable_percentage,
                    args.lr,
                    val_acc,
                    best_acc,
                    worst_acc,
                    abs(best_acc - worst_acc),
                    mask_filename,
                ]
            elif(args.sens_attribute == 'age'):
                best_acc = max(acc_age0_avg, acc_age1_avg, acc_age2_avg, acc_age3_avg, acc_age4_avg)
                worst_acc = min(acc_age0_avg, acc_age1_avg, acc_age2_avg, acc_age3_avg, acc_age4_avg)

                new_row = [
                    args.tuning_method,
                    trainable_percentage,
                    args.lr,
                    val_acc,
                    best_acc,
                    worst_acc,
                    abs(best_acc - worst_acc),
                    mask_filename,
                ]

        elif args.masking_vector is not None or args.subnetwork_mask_name is not None:
            new_row = [
                "Subnetwork_" + args.tuning_method,
                trainable_percentage,
                args.lr,
                val_acc,
                convert_numpy_array_to_string(mask),
            ]
        else:
            new_row = [
                args.tuning_method,
                trainable_percentage,
                args.lr,
                val_acc,
                os.path.join(args.vector_savepath, filename),
            ]

        # results_df.loc[len(results_df)] = new_row
        # results_df.to_csv(os.path.join(args.output_dir, args.results_df), index=False)

        # plot_changes(ckpt_path, base_model, args)

        if args.save_flag:
            print(
                "Saving Masking Vector at: {}".format(
                    os.path.join(args.vector_savepath, filename)
                )
            )

        # Obtaining the performance on test set
        print("Obtaining the performance on test set")
        if args.sens_attribute == "gender":
            (
                test_acc,
                test_male_acc,
                test_female_acc,
                test_loss,
                test_max_loss,
            ) = evaluate_fairness_gender(
                model,
                criterion,
                ece_criterion,
                data_loader_test,
                args=args,
                device=device,
            )
            print("\n")
            print("Test Male Accuracy: ", test_male_acc)
            print("Test Female Accuracy: ", test_female_acc)
        elif args.sens_attribute == "skin_type":
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
            if(args.age_type == 'multi'):
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
            elif(args.age_type == 'binary'):
                (
                    test_acc,
                    test_acc_type0,
                    test_acc_type1,
                    test_loss,
                    test_max_loss,
                ) = evaluate_fairness_age_binary(
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
            else:
                raise NotImplementedError("Age type not supported. Choose from 'multi' or 'binary'")
        else:
            raise NotImplementedError

        print("Test accuracy: ", test_acc)
        print("Test loss: ", round(torch.mean(test_loss).item(), 3))
        print("Test max loss: ", round(test_max_loss.item(), 3))

        # Add these results to CSV
        # columns=['Tuning Method','Train Percent','LR Scaler', 'Inner LR', 'Outer LR','Test Acc@1','Vector Path']

        # Here we are adding results on the test set
        if args.masking_vector_idx is not None:
            if(args.sens_attribute == 'gender'):
                new_row2 = [
                    args.tuning_method,
                    round(trainable_percentage, 3),
                    np.nan,
                    args.lr,
                    np.nan,
                    test_acc,
                    test_male_acc,
                    test_female_acc,
                    round(abs(test_male_acc - test_female_acc), 3),
                    mask_filename,
                ]
            else:
                if(args.sens_attribute == 'skin_type'):
                    best_acc = max(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4, test_acc_type5)
                    worst_acc = min(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4, test_acc_type5)
                elif(args.sens_attribute == 'age'):
                    if(args.age_type == 'multi'):
                        best_acc = max(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4)
                        worst_acc = min(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4)
                    elif(args.age_type == 'binary'):
                        best_acc = max(test_acc_type0, test_acc_type1)
                        worst_acc = min(test_acc_type0, test_acc_type1)
                    else:
                        raise NotImplementedError("Age type not supported. Choose from 'multi' or 'binary'")
                        
                new_row2 = [
                    args.tuning_method,
                    round(trainable_percentage, 3),
                    np.nan,
                    args.lr,
                    np.nan,
                    test_acc,
                    best_acc,
                    worst_acc,
                    round(abs(best_acc - worst_acc), 3),
                    mask_filename,
                ]
            
        elif args.masking_vector is not None or args.subnetwork_mask_name is not None:
            new_row2 = [
                "Subnetwork_" + args.tuning_method,
                round(trainable_percentage, 3),
                np.nan,
                args.lr,
                np.nan,
                test_acc,
                test_male_acc,
                test_female_acc,
                round(abs(test_male_acc - test_female_acc), 3),
                convert_numpy_array_to_string(mask),
            ]
        test_results_df.loc[len(test_results_df)] = new_row2

        print(
            "Saving test results df at: {}".format(
                os.path.join(args.output_dir, args.test_results_df)
            )
        )

        test_results_df.to_csv(
            os.path.join(args.output_dir, args.test_results_df), index=False
        )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.output_dir = os.path.join(os.getcwd(), args.model, args.dataset)
    args.results_df = (
        "Fairness_Fixed_Vectors_"
        + args.sens_attribute
        + "_"
        + args.tuning_method
        + "_"
        + args.model
        + ".csv"
    )
    args.test_results_df = (
        "Fairness_Test_set_results_"
        + args.sens_attribute
        + "_"
        + args.tuning_method
        + "_"
        + args.model
        + ".csv"
    )

    current_wd = os.getcwd()
    args.vector_savepath = os.path.join(
        current_wd,
        "saved_vectors",
        args.model,
        args.dataset,
        args.tuning_method + "_" + str(args.lr),
    )
    args.fig_savepath = os.path.join(args.output_dir, "plots/")

    args.binary_mask_savepath = os.path.join(
        current_wd,
        "Saved_Binary_Vectors_Subnetwork",
        args.model,
        args.dataset,
        args.tuning_method,
    )

    if (
        args.masking_vector_idx is None
        and args.masking_vector is None
        and args.subnetwork_mask_name is None
        and args.tuning_method != "fullft"
    ):
        args.save_flag = True
    else:
        args.save_flag = False

    main(args)
