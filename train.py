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


def main(args):

    if(args.save_flag):
        try:
            os.makedirs(args.vector_savepath)
        except:
            print("Directory exists")

    os.makedirs(args.fig_savepath, exist_ok=True)
    masking_idx = args.masking_vector_idx
    print("Masking Index: ", masking_idx)

    if(masking_idx is not None):
        mask_filename = os.path.join(args.exp_vector_path, 'random_vector_' + str(masking_idx) + '.npy')
        mask = np.load(mask_filename)
        print(mask)
    else:
        mask=None

    #if(args.masking_vector_idx is None):
    if(args.save_flag):
        files = os.listdir(args.vector_savepath)
        files = [f for f in files if re.match(args.tuning_method + '_' + r'vector_\d+.npy', f)]
    
        if(len(files) > 0):
            numbers = [int(re.search(r'\d+', f).group()) for f in files]
            vector_idx = max(numbers) if numbers else None
        else:
            vector_idx = 0

        print(vector_idx, type(vector_idx))

    #Making directory for saving checkpoints
    if args.output_dir:
        utils.mkdir(args.output_dir)
        utils.mkdir(os.path.join(args.output_dir, 'checkpoints'))
    
    try:
        results_df = pd.read_csv(os.path.join(args.output_dir, args.results_df))
        test_results_df = pd.read_csv(os.path.join(args.output_dir, args.test_results_df))
    except:
        #columns=['Tuning Method','Train Percent','LR Scaler', 'Inner LR', 'Outer LR','Test Acc@1','Vector Path']
        results_df = pd.DataFrame(columns=['Tuning Method','Train Percent','LR','Test Acc@1','Vector Path'])
        test_results_df = pd.DataFrame(columns=['Tuning Method','Train Percent','LR Scaler','Inner LR', 'Outer LR','Test Acc@1','Vector Path'])


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
    print(dataset.class_to_idx)

    args.class_to_idx = {value: key for key, value in dataset.class_to_idx.items()}
    
    print(args.class_to_idx)

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

    print("Creating model")

    model = utils.get_timm_model(args.model, num_classes=args.num_classes)
    base_model = model

    print("TUNING METHOD: ", args.tuning_method)

    if(args.tuning_method != 'fullft'):

        if(args.tuning_method == 'lora'):
            model = utils.create_lora_model(model, block_mask=mask)

        else:
            masking_vector = utils.get_masked_model(model, args.tuning_method, mask=list(mask))
            print("MASKING VECTOR: ", masking_vector)


    # Save the Masking Vector
    #if(args.masking_vector_idx is None):
    # if(args.save_flag):
    #     vector_idx += 1
    #     print("VECTOR INDEX: ", vector_idx)
    #     filename = args.tuning_method + '_' + 'vector_' + str(vector_idx) + '.npy'

        # if(args.tuning_method != 'fullft'):
            
        #     with open(os.path.join(args.vector_savepath, filename), 'wb') as f:
        #         np.save(f, np.array(masking_vector))

    #Add Linear Layer
    # linear_layer = nn.Linear(model.head.in_features, args.num_classes)
    # torch.nn.init.zeros_(linear_layer.weight)
    # model.head = linear_layer

    #Check Tunable Params
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
    optimizer = get_optimizer(args, parameters)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    #LR Scheduler
    lr_scheduler = get_lr_scheduler(args, optimizer)

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
            optimizer.load_state_dict(checkpoint["optimizer"])
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
            evaluate(model_ema, criterion, ece_criterion, data_loader_val, args=args, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, ece_criterion, data_loader_val, args=args, device=device)
        return

    if(args.disable_training):
        print("Training Process Skipped")
    else:
        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_one_epoch(model, criterion, ece_criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
            lr_scheduler.step()
            val_acc, val_loss = evaluate(model, criterion, ece_criterion, data_loader_val, args=args, device=device)
            if model_ema:
                val_acc, val_loss = evaluate(model_ema, criterion, ece_criterion, data_loader_val, args=args, device=device, log_suffix="EMA")
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
                #utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoints', f"model_{epoch}.pth"))7
                ckpt_path = os.path.join(args.output_dir, 'checkpoints', "checkpoint_" + args.tuning_method + ".pth")
                utils.save_on_master(checkpoint, ckpt_path)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
        #print(f"Training time {total_time_str}")

        # Add all the information to the results_df
        # 'Tuning Method','Train Percent','LR','Test Acc@1','Vector Path'
        if(args.masking_vector_idx is not None):
            new_row = [args.tuning_method, trainable_percentage, args.lr, val_acc, mask_filename]
        else:
            new_row = [args.tuning_method, trainable_percentage, args.lr, val_acc, os.path.join(args.vector_savepath, filename)]

        results_df.loc[len(results_df)] = new_row
        results_df.to_csv(os.path.join(args.output_dir, args.results_df), index=False)

        #plot_changes(ckpt_path, base_model, args)

        if(args.save_flag):
            print("Saving Masking Vector at: {}".format(os.path.join(args.vector_savepath, filename)))

        print("Saving results df at: {}".format(os.path.join(args.output_dir, args.results_df)))

        # Obtaining the performance on test set
        test_acc, test_loss = evaluate(model, criterion, ece_criterion, data_loader_test, args=args, device=device)
        print("Test accuracy: ", test_acc)
        print("Test loss: ", test_loss)

        # Add these results to CSV
        #columns=['Tuning Method','Train Percent','LR Scaler', 'Inner LR', 'Outer LR','Test Acc@1','Vector Path']
        new_row2 = [args.tuning_method, trainable_percentage, np.nan, args.lr, np.nan, test_acc, mask_filename]
        test_results_df.loc[len(test_results_df)] = new_row2
        test_results_df.to_csv(os.path.join(args.output_dir, args.test_results_df), index=False)


if __name__ == "__main__":

    args = get_args_parser().parse_args()
    args.output_dir = os.path.join(os.getcwd(), args.model, args.dataset)
    args.results_df = 'Fixed_Vectors_' + args.tuning_method + '_' + args.model + '.csv'
    args.test_results_df = 'Test_set_results_' + args.tuning_method + '_' + args.model + '.csv'

    current_wd = os.getcwd()
    args.vector_savepath = os.path.join(current_wd, 'saved_vectors', args.model, args.dataset, args.tuning_method + '_' + str(args.lr))
    args.fig_savepath = os.path.join(args.output_dir, 'plots/')
    
    if(args.masking_vector_idx is None and args.tuning_method != 'fullft'):
        args.save_flag = True
    else:
        args.save_flag = False

    main(args)