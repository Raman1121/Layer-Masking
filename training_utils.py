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
from data import HAM10000, fitzpatrick, papila, ol3i, oasis
from torch.utils.data.sampler import SubsetRandomSampler
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

######################################################################


def train_one_epoch(
    model,
    criterion,
    ece_criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
    **kwargs,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
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
        # auc = utils.auc(output, target, pos_label=kwargs['pos_label'])
        auc_dict = utils.roc_auc_score_multiclass(output, target)
        auc = sum(auc_dict.keys()) / len(auc_dict)

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(
            ece_loss=ece_loss.item(), lr=optimizer.param_groups[0]["lr"]
        )
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["auc"].update(auc, n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def train_one_epoch_fairness(
    model,
    criterion,
    ece_criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
    **kwargs,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))


    header = f"Epoch: [{epoch}]"
    for i, (image, target, sens_attr) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)

            # if(args.num_classes == 2):
            #     #We are using Binary Cross Entropy Loss
            #     output = output.view(-1)

            #print(output.dtype, target.dtype)
            #print(output, target)
            loss = torch.mean(criterion(output, target))

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

        # acc1, acc5 = utils.accuracy(output, target, topk=(1, args.num_classes))
        # auc = utils.auc(output, target, pos_label=kwargs['pos_label'])
        if args.sens_attribute == "gender":
            acc1, acc_male, acc_female = utils.accuracy_by_gender(
                output, target, sens_attr, topk=(1, args.num_classes)
            )
            acc1 = acc1[0]
            acc_male = acc_male[0]
            acc_female = acc_female[0]

        elif args.sens_attribute == "skin_type":
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

        elif args.sens_attribute == "age":
            if(args.age_type == 'multi'):
                acc1, res_type0, res_type1, res_type2, res_type3, res_type4 = utils.accuracy_by_age(
                    output, target, sens_attr, topk=(1,)
                )
                acc1 = acc1[0]
                acc_type0 = res_type0[0]
                acc_type1 = res_type1[0]
                acc_type2 = res_type2[0]
                acc_type3 = res_type3[0]
                acc_type4 = res_type4[0]
            elif(args.age_type == 'binary'):
                acc1, res_type0, res_type1 = utils.accuracy_by_age_binary(
                    output, target, sens_attr, topk=(1,)
                )
                acc1 = acc1[0]
                acc_type0 = res_type0[0]
                acc_type1 = res_type1[0]
            else:
                raise NotImplementedError("Age type not implemented")
        else:
            raise NotImplementedError

        #auc_dict = utils.roc_auc_score_multiclass(output, target)
        #auc = sum(auc_dict.keys()) / len(auc_dict)

        batch_size = image.shape[0]

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(
            ece_loss=ece_loss.item(), lr=optimizer.param_groups[0]["lr"]
        )
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

        if args.sens_attribute == "gender":
            metric_logger.meters["acc1_male"].update(acc_male.item(), n=batch_size)
            metric_logger.meters["acc1_female"].update(acc_female.item(), n=batch_size)
        elif args.sens_attribute == "skin_type":
            if acc_type0 is not np.nan:
                metric_logger.meters["acc_type0"].update(acc_type0.item(), n=batch_size)
            else:
                metric_logger.meters["acc_type0"].update(0.0, n=batch_size)

            if acc_type1 is not np.nan: 
                metric_logger.meters["acc_type1"].update(acc_type1.item(), n=batch_size)
            else:
                metric_logger.meters["acc_type1"].update(0.0, n=batch_size)
            
            if acc_type2 is not np.nan:
                metric_logger.meters["acc_type2"].update(acc_type2.item(), n=batch_size)
            else:
                metric_logger.meters["acc_type2"].update(0.0, n=batch_size)
            
            if acc_type3 is not np.nan:
                metric_logger.meters["acc_type3"].update(acc_type3.item(), n=batch_size)
            else:
                metric_logger.meters["acc_type3"].update(0.0, n=batch_size)

            if acc_type4 is not np.nan:
                metric_logger.meters["acc_type4"].update(acc_type4.item(), n=batch_size)
            else:
                metric_logger.meters["acc_type4"].update(0.0, n=batch_size)
            
            if acc_type5 is not np.nan:
                metric_logger.meters["acc_type5"].update(acc_type5.item(), n=batch_size)
            else:
                metric_logger.meters["acc_type5"].update(0.0, n=batch_size)

        elif args.sens_attribute == "age":
            if(args.age_type == 'multi'):
                if acc_type0 is not np.nan:
                    metric_logger.meters["acc_Age0"].update(acc_type0.item(), n=batch_size)
                else:
                    metric_logger.meters["acc_Age0"].update(0.0, n=batch_size)

                if acc_type1 is not np.nan: 
                    metric_logger.meters["acc_Age1"].update(acc_type1.item(), n=batch_size)
                else:
                    metric_logger.meters["acc_Age1"].update(0.0, n=batch_size)
                
                if acc_type2 is not np.nan:
                    metric_logger.meters["acc_Age2"].update(acc_type2.item(), n=batch_size)
                else:
                    metric_logger.meters["acc_Age2"].update(0.0, n=batch_size)
                
                if acc_type3 is not np.nan:
                    metric_logger.meters["acc_Age3"].update(acc_type3.item(), n=batch_size)
                else:
                    metric_logger.meters["acc_Age3"].update(0.0, n=batch_size)

                if acc_type4 is not np.nan:
                    metric_logger.meters["acc_Age4"].update(acc_type4.item(), n=batch_size)
                else:
                    metric_logger.meters["acc_Age4"].update(0.0, n=batch_size)
            elif(args.age_type == 'binary'):
                if acc_type0 is not np.nan:
                    metric_logger.meters["acc_Age0"].update(acc_type0.item(), n=batch_size)
                else:
                    metric_logger.meters["acc_Age0"].update(0.0, n=batch_size)

                if acc_type1 is not np.nan: 
                    metric_logger.meters["acc_Age1"].update(acc_type1.item(), n=batch_size)
                else:
                    metric_logger.meters["acc_Age1"].update(0.0, n=batch_size)
            else:
                raise NotImplementedError("Age type not implemented")

        # metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        # metric_logger.meters["auc"].update(auc, n=batch_size)
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
    for n, p in model.named_parameters():
        if "attn" in n:
            attn_params.append(p)

    return attn_params


def get_fast_params(model, args):
    fast_params = []
    param_names= []

    # Get fast parameters from each module
    if args.tuning_method == "tune_attention_blocks_random":
        for name, module in model.named_modules():
            if "fw" in module.__class__.__name__:
                if("attn" in name):
                    for n, param in module.named_parameters():
                        fast_params.append(param)
                        param_names.append(n)

    elif args.tuning_method == "tune_layernorm_blocks_random":
        for name, module in model.named_modules():
            if "fw" in module.__class__.__name__:
                if "norm" in name:
                    for n, param in module.named_parameters():
                        fast_params.append(param)
                        param_names.append(n)

    elif args.tuning_method == "tune_blocks_random":
        for name, module in model.named_modules():
            if "fw" in module.__class__.__name__:
                for n, param in module.named_parameters():
                    fast_params.append(param)
                    param_names.append(n)

    return fast_params, param_names


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
        mask_dict["mask_el_" + str(i)].append(round(mask[i], 6))

    return mask_dict


def plot_binary_mask(binary_mask):
    binary_mask = binary_mask.detach().cpu().numpy()
    heatmap_data = binary_mask.reshape((1, -1))
    fig, ax = plt.subplots(figsize=(len(binary_mask) * 2, 2))
    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        cbar=False,
        ax=ax,
        square=True,
        linecolor="black",
        linewidths=1,
    )
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
    axs[-1].set_xlabel("Training Steps")
    # axs[-1].set_xticks()
    fig.suptitle("Change in Mask Values during Training")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the plot as a PNG image
    plot_path = (
        "Mask_plot_" + args.dataset + "_MaskGen_" + args.mask_gen_method + ".png"
    )
    plt.savefig(os.path.join(args.fig_savepath, plot_path))

    if args.wandb_logging:
        # wandb.log({"Mask Params during Training": fig})
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


def meta_train_one_epoch(
    model,
    criterion,
    ece_criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
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
        metric_logger.update(
            ece_loss=ece_loss.item(), lr=optimizer.param_groups[0]["lr"]
        )
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(
    model,
    criterion,
    ece_criterion,
    data_loader,
    device,
    args,
    print_freq=100,
    log_suffix="",
    **kwargs,
):
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
            # auc = utils.auc(output, target, pos_label=kwargs['pos_label'])
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

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    return metric_logger.acc1.global_avg, loss, auc


def evaluate_fairness_gender(
    model,
    criterion,
    ece_criterion,
    data_loader,
    device,
    args,
    print_freq=100,
    log_suffix="",
    **kwargs,
):
    print("EVALUATING")
    model.eval()

    assert args.sens_attribute == "gender"

    total_loss_male = 0.0
    total_loss_female = 0.0
    num_male = 0
    num_female = 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target, sens_attr in metric_logger.log_every(
            data_loader, print_freq, header
        ):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # sens_attr = sens_attr.to(device, non_blocking=True)

            output = model(image)
            loss = criterion(output, target)
            ece_loss = ece_criterion(output, target)

            indexes_males = [
                index for index, gender in enumerate(sens_attr) if gender == "M"
            ]
            indexes_females = [
                index for index, gender in enumerate(sens_attr) if gender == "F"
            ]

            loss_male = [loss[index] for index in indexes_males]
            loss_female = [loss[index] for index in indexes_females]

            total_loss_male = sum(loss_male)
            total_loss_female = sum(loss_female)

            # num_male += torch.sum(sens_attr == 'M').item()
            # num_female += torch.sum(sens_attr == 'F').item()

            num_male += sens_attr.count("M")
            num_female += sens_attr.count("F")

            avg_loss_male = total_loss_male / num_male if num_male > 0 else 0.0
            avg_loss_female = total_loss_female / num_female if num_female > 0 else 0.0

            # Take the maximum of the two losses
            max_val_loss = max(avg_loss_male, avg_loss_female)
            diff_loss = torch.abs(avg_loss_male - avg_loss_female)
            acc1, acc_male, acc_female = utils.accuracy_by_gender(
                output, target, sens_attr, topk=(1,)
            )
            acc1 = acc1[0]
            acc_male = acc_male[0]
            acc_female = acc_female[0]

            acc1_orig, acc5 = utils.accuracy(output, target, topk=(1, args.num_classes))
            auc = 0

            batch_size = image.shape[0]
            metric_logger.update(loss=torch.mean(loss).item())
            metric_logger.update(ece_loss=ece_loss.item())
            metric_logger.update(max_val_loss=max_val_loss)
            metric_logger.update(diff_loss=diff_loss)
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc1_male"].update(acc_male.item(), n=batch_size)
            metric_logger.meters["acc1_female"].update(acc_female.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["auc"].update(auc, n=batch_size)
            metric_logger.meters["max_val_loss"].update(max_val_loss, n=batch_size)
            metric_logger.meters["diff_loss"].update(diff_loss, n=batch_size)
            num_processed_samples += batch_size

    # gather the stats from all processes
    # num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    # if (
    #     hasattr(data_loader.dataset, "__len__")
    #     and len(data_loader.dataset) != num_processed_samples
    #     and torch.distributed.get_rank() == 0
    # ):
    #     # See FIXME above
    #     warnings.warn(
    #         f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
    #         "samples were used for the validation, which might bias the results. "
    #         "Try adjusting the batch size and / or the world size. "
    #         "Setting the world size to 1 is always a safe bet."
    #     )

    metric_logger.synchronize_between_processes()

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} Max Loss {metric_logger.max_val_loss.global_avg:.3f} Diff Loss {metric_logger.diff_loss.global_avg:.3f}"
    )

    acc_avg = metric_logger.acc1.global_avg
    male_acc_avg = metric_logger.acc1_male.global_avg
    female_acc_avg = metric_logger.acc1_female.global_avg

    return (
        round(acc_avg, 3),
        round(male_acc_avg, 3),
        round(female_acc_avg, 3),
        loss,
        max_val_loss,
    )


def evaluate_fairness_skin_type(
    model,
    criterion,
    ece_criterion,
    data_loader,
    device,
    args,
    print_freq=100,
    log_suffix="",
    **kwargs,
):
    print("EVALUATING")
    model.eval()

    assert args.sens_attribute == "skin_type"

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

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target, sens_attr in metric_logger.log_every(
            data_loader, print_freq, header
        ):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # sens_attr = sens_attr.to(device, non_blocking=True)

            output = model(image)
            loss = criterion(output, target)
            ece_loss = ece_criterion(output, target)

            loss_type1 = torch.mean(loss[sens_attr == 0])
            loss_type2 = torch.mean(loss[sens_attr == 1])
            loss_type3 = torch.mean(loss[sens_attr == 2])
            loss_type4 = torch.mean(loss[sens_attr == 3])
            loss_type5 = torch.mean(loss[sens_attr == 4])
            loss_type6 = torch.mean(loss[sens_attr == 5])

            total_loss_type1 += loss_type1.item()
            total_loss_type2 += loss_type2.item()
            total_loss_type3 += loss_type3.item()
            total_loss_type4 += loss_type4.item()
            total_loss_type5 += loss_type5.item()
            total_loss_type6 += loss_type6.item()

            num_type1 += torch.sum(sens_attr == 0).item()
            num_type2 += torch.sum(sens_attr == 1).item()
            num_type3 += torch.sum(sens_attr == 2).item()
            num_type4 += torch.sum(sens_attr == 3).item()
            num_type5 += torch.sum(sens_attr == 4).item()
            num_type6 += torch.sum(sens_attr == 5).item()

            total_losses = [
                total_loss_type1,
                total_loss_type2,
                total_loss_type3,
                total_loss_type4,
                total_loss_type5,
                total_loss_type6,
            ]
            num_samples = [
                num_type1,
                num_type2,
                num_type3,
                num_type4,
                num_type5,
                num_type6,
            ]

            avg_losses = []
            for total_loss, num in zip(total_losses, num_samples):
                avg_loss = total_loss / num if num > 0 else 0.0
                avg_losses.append(avg_loss)

            avg_loss_type1 = avg_losses[0]
            avg_loss_type2 = avg_losses[1]
            avg_loss_type3 = avg_losses[2]
            avg_loss_type4 = avg_losses[3]
            avg_loss_type5 = avg_losses[4]
            avg_loss_type6 = avg_losses[5]

            # Take the maximum and minimum of all the losses
            max_val_loss = torch.tensor(
                max(
                    avg_loss_type1,
                    avg_loss_type2,
                    avg_loss_type3,
                    avg_loss_type4,
                    avg_loss_type5,
                    avg_loss_type6,
            ))
            min_val_loss = torch.tensor(
                min(
                    avg_loss_type1,
                    avg_loss_type2,
                    avg_loss_type3,
                    avg_loss_type4,
                    avg_loss_type5,
                    avg_loss_type6,
            ))

            # Take the difference between the greatest and the smallest loss
            #print("max_val_loss", max_val_loss)
            #print(type(max_val_loss))
            diff_loss = torch.abs(max_val_loss - min_val_loss)

            
            acc1, res_type0, res_type1, res_type2, res_type3, res_type4, res_type5 = utils.accuracy_by_skin_type(
                output, target, sens_attr, topk=(1,), num_skin_types=args.num_skin_types
            )
            # acc1, acc_male, acc_female = utils.accuracy_by_gender(output, target, sens_attr, topk=(1,))
            acc1 = acc1[0]

            try:
                acc_type0 = res_type0[0]
            except:
                acc_type0 = torch.tensor(0.0)
            
            try:
                acc_type1 = res_type1[0]
            except:
                acc_type1 = torch.tensor(0.0)
            
            try:
                acc_type2 = res_type2[0]
            except:
                acc_type2 = torch.tensor(0.0)
            
            try:
                acc_type3 = res_type3[0]
            except:
                acc_type3 = torch.tensor(0.0)

            try:
                acc_type4 = res_type4[0]
            except:
                acc_type4 = torch.tensor(0.0)
            
            try:
                acc_type5 = res_type5[0]
            except:
                acc_type5 = torch.tensor(0.0)

            acc1_orig, acc5 = utils.accuracy(output, target, topk=(1, args.num_classes))
            auc = 0

            batch_size = image.shape[0]
            metric_logger.update(loss=torch.mean(loss).item())
            metric_logger.update(ece_loss=ece_loss.item())
            metric_logger.update(max_val_loss=max_val_loss)
            metric_logger.update(diff_loss=diff_loss)
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc_type0"].update(acc_type0.item(), n=batch_size)
            metric_logger.meters["acc_type1"].update(acc_type1.item(), n=batch_size)
            metric_logger.meters["acc_type2"].update(acc_type2.item(), n=batch_size)
            metric_logger.meters["acc_type3"].update(acc_type3.item(), n=batch_size)
            metric_logger.meters["acc_type4"].update(acc_type4.item(), n=batch_size)
            metric_logger.meters["acc_type5"].update(acc_type5.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["auc"].update(auc, n=batch_size)
            metric_logger.meters["max_val_loss"].update(max_val_loss, n=batch_size)
            metric_logger.meters["diff_loss"].update(diff_loss, n=batch_size)
            num_processed_samples += batch_size

    # gather the stats from all processes
    # num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    # if (
    #     hasattr(data_loader.dataset, "__len__")
    #     and len(data_loader.dataset) != num_processed_samples
    #     and torch.distributed.get_rank() == 0
    # ):
    #     # See FIXME above
    #     warnings.warn(
    #         f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
    #         "samples were used for the validation, which might bias the results. "
    #         "Try adjusting the batch size and / or the world size. "
    #         "Setting the world size to 1 is always a safe bet."
    #     )

    metric_logger.synchronize_between_processes()

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} Max Loss {metric_logger.max_val_loss.global_avg:.3f} Diff Loss {metric_logger.diff_loss.global_avg:.3f}"
    )
    # return metric_logger.acc1.global_avg, loss, max_val_loss, metric_logger.acc1_male.global_avg, metric_logger.acc1_female.global_avg
    # TODO: Use a different return statement here

    acc_avg = metric_logger.acc1.global_avg
    acc_type0_avg = metric_logger.acc_type0.global_avg
    acc_type1_avg = metric_logger.acc_type1.global_avg
    acc_type2_avg = metric_logger.acc_type2.global_avg
    acc_type3_avg = metric_logger.acc_type3.global_avg
    acc_type4_avg = metric_logger.acc_type4.global_avg
    acc_type5_avg = metric_logger.acc_type5.global_avg

    return (
        round(acc_avg, 3),
        round(acc_type0_avg, 3),
        round(acc_type1_avg, 3),
        round(acc_type2_avg, 3),
        round(acc_type3_avg, 3),
        round(acc_type4_avg, 3),
        round(acc_type5_avg, 3),
        loss,
        max_val_loss,
    )

def evaluate_fairness_age(
    model,
    criterion,
    ece_criterion,
    data_loader,
    device,
    args,
    print_freq=100,
    log_suffix="",
    **kwargs,
):
    print("EVALUATING")
    model.eval()

    assert args.sens_attribute == "age"

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

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target, sens_attr in metric_logger.log_every(
            data_loader, print_freq, header
        ):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # sens_attr = sens_attr.to(device, non_blocking=True)

            output = model(image)
            loss = criterion(output, target)
            ece_loss = ece_criterion(output, target)

            loss_type1 = torch.mean(loss[sens_attr == 0])
            loss_type2 = torch.mean(loss[sens_attr == 1])
            loss_type3 = torch.mean(loss[sens_attr == 2])
            loss_type4 = torch.mean(loss[sens_attr == 3])
            loss_type5 = torch.mean(loss[sens_attr == 4])

            total_loss_type1 += loss_type1.item()
            total_loss_type2 += loss_type2.item()
            total_loss_type3 += loss_type3.item()
            total_loss_type4 += loss_type4.item()
            total_loss_type5 += loss_type5.item()

            num_type1 += torch.sum(sens_attr == 0).item()
            num_type2 += torch.sum(sens_attr == 1).item()
            num_type3 += torch.sum(sens_attr == 2).item()
            num_type4 += torch.sum(sens_attr == 3).item()
            num_type5 += torch.sum(sens_attr == 4).item()

            total_losses = [
                total_loss_type1,
                total_loss_type2,
                total_loss_type3,
                total_loss_type4,
                total_loss_type5,
            ]
            num_samples = [
                num_type1,
                num_type2,
                num_type3,
                num_type4,
                num_type5,
            ]

            avg_losses = []
            for total_loss, num in zip(total_losses, num_samples):
                avg_loss = total_loss / num if num > 0 else 0.0
                avg_losses.append(avg_loss)

            avg_loss_type1 = avg_losses[0]
            avg_loss_type2 = avg_losses[1]
            avg_loss_type3 = avg_losses[2]
            avg_loss_type4 = avg_losses[3]
            avg_loss_type5 = avg_losses[4]

            # Take the maximum and minimum of all the losses
            max_val_loss = torch.tensor(
                max(
                    avg_loss_type1,
                    avg_loss_type2,
                    avg_loss_type3,
                    avg_loss_type4,
                    avg_loss_type5,
            ))
            min_val_loss = torch.tensor(
                min(
                    avg_loss_type1,
                    avg_loss_type2,
                    avg_loss_type3,
                    avg_loss_type4,
                    avg_loss_type5,
            ))

            # Take the difference between the greatest and the smallest loss
            #print("max_val_loss", max_val_loss)
            #print(type(max_val_loss))
            diff_loss = torch.abs(max_val_loss - min_val_loss)

            
            acc1, res_type0, res_type1, res_type2, res_type3, res_type4 = utils.accuracy_by_age(
                output, target, sens_attr, topk=(1,)
            )
            
            acc1 = acc1[0]

            try:
                acc_type0 = res_type0[0]
            except:
                acc_type0 = torch.tensor(0.0)
            
            try:
                acc_type1 = res_type1[0]
            except:
                acc_type1 = torch.tensor(0.0)
            
            try:
                acc_type2 = res_type2[0]
            except:
                acc_type2 = torch.tensor(0.0)
            
            try:
                acc_type3 = res_type3[0]
            except:
                acc_type3 = torch.tensor(0.0)

            try:
                acc_type4 = res_type4[0]
            except:
                acc_type4 = torch.tensor(0.0)
            

            acc1_orig, acc5 = utils.accuracy(output, target, topk=(1, args.num_classes))
            auc = 0

            batch_size = image.shape[0]
            metric_logger.update(loss=torch.mean(loss).item())
            metric_logger.update(ece_loss=ece_loss.item())
            metric_logger.update(max_val_loss=max_val_loss)
            metric_logger.update(diff_loss=diff_loss)
            metric_logger.meters["acc1"].update(acc1_orig.item(), n=batch_size)
            metric_logger.meters["acc_Age0"].update(acc_type0.item(), n=batch_size)
            metric_logger.meters["acc_Age1"].update(acc_type1.item(), n=batch_size)
            metric_logger.meters["acc_Age2"].update(acc_type2.item(), n=batch_size)
            metric_logger.meters["acc_Age3"].update(acc_type3.item(), n=batch_size)
            metric_logger.meters["acc_Age4"].update(acc_type4.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["auc"].update(auc, n=batch_size)
            metric_logger.meters["max_val_loss"].update(max_val_loss, n=batch_size)
            metric_logger.meters["diff_loss"].update(diff_loss, n=batch_size)
            num_processed_samples += batch_size

    # gather the stats from all processes
    # num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    # if (
    #     hasattr(data_loader.dataset, "__len__")
    #     and len(data_loader.dataset) != num_processed_samples
    #     and torch.distributed.get_rank() == 0
    # ):
    #     # See FIXME above
    #     warnings.warn(
    #         f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
    #         "samples were used for the validation, which might bias the results. "
    #         "Try adjusting the batch size and / or the world size. "
    #         "Setting the world size to 1 is always a safe bet."
    #     )

    metric_logger.synchronize_between_processes()

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} Max Loss {metric_logger.max_val_loss.global_avg:.3f} Diff Loss {metric_logger.diff_loss.global_avg:.3f}"
    )
    # return metric_logger.acc1.global_avg, loss, max_val_loss, metric_logger.acc1_male.global_avg, metric_logger.acc1_female.global_avg
    # TODO: Use a different return statement here

    acc_avg = metric_logger.acc1.global_avg
    acc_age0_avg = metric_logger.acc_Age0.global_avg
    acc_age1_avg = metric_logger.acc_Age1.global_avg
    acc_age2_avg = metric_logger.acc_Age2.global_avg
    acc_age3_avg = metric_logger.acc_Age3.global_avg
    acc_age4_avg = metric_logger.acc_Age4.global_avg

    return (
        round(acc_avg, 3),
        round(acc_age0_avg, 3),
        round(acc_age1_avg, 3),
        round(acc_age2_avg, 3),
        round(acc_age3_avg, 3),
        round(acc_age4_avg, 3),
        loss,
        max_val_loss,
    )

def evaluate_fairness_age_binary(
    model,
    criterion,
    ece_criterion,
    data_loader,
    device,
    args,
    print_freq=100,
    log_suffix="",
    **kwargs,
):
    print("EVALUATING")
    model.eval()

    assert args.sens_attribute == "age"

    total_loss_type1 = 0.0
    total_loss_type2 = 0.0

    num_type1 = 0
    num_type2 = 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target, sens_attr in metric_logger.log_every(
            data_loader, print_freq, header
        ):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # sens_attr = sens_attr.to(device, non_blocking=True)

            output = model(image)
            loss = criterion(output, target)
            ece_loss = ece_criterion(output, target)

            loss_type1 = torch.mean(loss[sens_attr == 0])
            loss_type2 = torch.mean(loss[sens_attr == 1])
            

            total_loss_type1 += loss_type1.item()
            total_loss_type2 += loss_type2.item()
            

            num_type1 += torch.sum(sens_attr == 0).item()
            num_type2 += torch.sum(sens_attr == 1).item()
            
            total_losses = [
                total_loss_type1,
                total_loss_type2,
                
            ]
            num_samples = [
                num_type1,
                num_type2,
            ]

            avg_losses = []
            for total_loss, num in zip(total_losses, num_samples):
                avg_loss = total_loss / num if num > 0 else 0.0
                avg_losses.append(avg_loss)

            avg_loss_type1 = avg_losses[0]
            avg_loss_type2 = avg_losses[1]

            # Take the maximum and minimum of all the losses
            max_val_loss = torch.tensor(
                max(
                    avg_loss_type1,
                    avg_loss_type2,
                    
            ))
            min_val_loss = torch.tensor(
                min(
                    avg_loss_type1,
                    avg_loss_type2,
                    
            ))

            diff_loss = torch.abs(max_val_loss - min_val_loss)

            
            acc1, res_type0, res_type1 = utils.accuracy_by_age_binary(
                output, target, sens_attr, topk=(1,)
            )
            
            acc1 = acc1[0]

            try:
                acc_type0 = res_type0[0]
            except:
                acc_type0 = torch.tensor(0.0)
            
            try:
                acc_type1 = res_type1[0]
            except:
                acc_type1 = torch.tensor(0.0)
            
            acc1_orig, acc5 = utils.accuracy(output, target, topk=(1, args.num_classes))
            auc = 0

            batch_size = image.shape[0]
            metric_logger.update(loss=torch.mean(loss).item())
            metric_logger.update(ece_loss=ece_loss.item())
            metric_logger.update(max_val_loss=max_val_loss)
            metric_logger.update(diff_loss=diff_loss)
            metric_logger.meters["acc1"].update(acc1_orig.item(), n=batch_size)
            metric_logger.meters["acc_Age0"].update(acc_type0.item(), n=batch_size)
            metric_logger.meters["acc_Age1"].update(acc_type1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["auc"].update(auc, n=batch_size)
            metric_logger.meters["max_val_loss"].update(max_val_loss, n=batch_size)
            metric_logger.meters["diff_loss"].update(diff_loss, n=batch_size)
            num_processed_samples += batch_size

    metric_logger.synchronize_between_processes()

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} Max Loss {metric_logger.max_val_loss.global_avg:.3f} Diff Loss {metric_logger.diff_loss.global_avg:.3f}"
    )
    # return metric_logger.acc1.global_avg, loss, max_val_loss, metric_logger.acc1_male.global_avg, metric_logger.acc1_female.global_avg
    # TODO: Use a different return statement here

    acc_avg = metric_logger.acc1.global_avg
    acc_age0_avg = metric_logger.acc_Age0.global_avg
    acc_age1_avg = metric_logger.acc_Age1.global_avg
    

    return (
        round(acc_avg, 3),
        round(acc_age0_avg, 3),
        round(acc_age1_avg, 3),
        loss,
        max_val_loss,
    )


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    )
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

    if args.dataset != "CIFAR10" and args.dataset != "CIFAR100":
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

        if args.dataset == "CIFAR10":
            dataset = torchvision.datasets.CIFAR10(
                root=args.dataset_basepath,
                train=True,
                download=True,
                transform=transform,
            )

            num_train = len(dataset)
            indices = list(range(num_train))
            split = int(np.floor(args.val_split * num_train))

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

        elif args.dataset == "CIFAR100":
            dataset = torchvision.datasets.CIFAR100(
                root=args.dataset_basepath,
                train=True,
                download=True,
                transform=transform,
            )

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
    if args.dataset != "CIFAR10" and args.dataset != "CIFAR100":
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
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
            )
        if args.dataset == "CIFAR10":
            dataset_val = torch.utils.data.Subset(dataset, valid_idx)
            dataset_test = torchvision.datasets.CIFAR10(
                root=args.dataset_basepath,
                train=False,
                download=True,
                transform=preprocessing,
            )
        elif args.dataset == "CIFAR100":
            dataset_val = torch.utils.data.Subset(dataset, valid_idx)
            dataset_test = torchvision.datasets.CIFAR100(
                root=args.dataset_basepath,
                train=False,
                download=True,
                transform=preprocessing,
            )
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
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False
        )
    else:
        if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            # valid_sampler = torch.utils.data.SequentialSampler(dataset_val)
            valid_sampler = torch.utils.data.RandomSampler(dataset_val)

        # test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_sampler = torch.utils.data.RandomSampler(dataset_test)

    return (
        dataset,
        dataset_val,
        dataset_test,
        train_sampler,
        valid_sampler,
        test_sampler,
    )


def load_fairness_data(args, df, df_val, df_test):
    print("Loading fairness data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()

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

        if args.dataset == "HAM10000":
            dataset = HAM10000.HAM10000Dataset(df, args.sens_attribute, transform)
        elif args.dataset == "fitzpatrick":
            args.num_skin_types = 6
            dataset = fitzpatrick.FitzpatrickDataset(df, transform)
        elif args.dataset == "papila":
            dataset = papila.PapilaDataset(df, args.sens_attribute, transform)
        elif args.dataset == "ol3i":
            dataset = ol3i.OL3IDataset(df, args.sens_attribute, transform, args.age_type)
        elif args.dataset == "oasis":
            dataset = oasis.OASISDataset(df, args.sens_attribute, transform, args.age_type)
        else:
            raise NotImplementedError("{} dataset not implemented for training".format(args.dataset))

    print("Took", time.time() - st)
    print("Loading validation and test data")

    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_val from {cache_path}")
        dataset_val, _ = torch.load(cache_path)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            transform_eval = weights.transforms()
        else:
            transform_eval = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
            )
        if args.dataset == "HAM10000":
            dataset_val = HAM10000.HAM10000Dataset(df_val, args.sens_attribute, transform_eval)
            dataset_test = HAM10000.HAM10000Dataset(df_test, args.sens_attribute, transform_eval)
        elif args.dataset == "fitzpatrick":
            dataset_val = fitzpatrick.FitzpatrickDataset(df_val, transform_eval)
            dataset_test = fitzpatrick.FitzpatrickDataset(df_test, transform_eval)
        elif args.dataset == "papila":
            dataset_val = papila.PapilaDataset(df_val, args.sens_attribute, transform_eval)
            dataset_test = papila.PapilaDataset(df_test, args.sens_attribute, transform_eval)
        elif args.dataset == "ol3i":
            dataset_val = ol3i.OL3IDataset(df_val, args.sens_attribute, transform_eval, args.age_type)
            dataset_test = ol3i.OL3IDataset(df_test, args.sens_attribute, transform_eval, args.age_type)
        elif args.dataset == "oasis":
            dataset_val = oasis.OASISDataset(df_val, args.sens_attribute, transform_eval, args.age_type)
            dataset_test = oasis.OASISDataset(df_test, args.sens_attribute, transform_eval, args.age_type)
        else:
            raise NotImplementedError("{} dataset not implemented for either validation or test".format(args.dataset))

    print("Creating data loaders")

    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False
        )

    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        valid_sampler = torch.utils.data.RandomSampler(dataset_val)
        test_sampler = torch.utils.data.RandomSampler(dataset_test)

    return (
        dataset,
        dataset_val,
        dataset_test,
        train_sampler,
        valid_sampler,
        test_sampler,
    )


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
            parameters,
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(
            f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported."
        )

    return optimizer


def get_lr_scheduler(args, optimizer):
    args.lr_scheduler = args.lr_scheduler.lower()

    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_gamma
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs],
        )
    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler


def get_mixup_transforms(args):
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomMixup(args.num_classes, p=1.0, alpha=args.mixup_alpha)
        )
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomCutmix(args.num_classes, p=1.0, alpha=args.cutmix_alpha)
        )

    return mixup_transforms


def get_model_ema(model_without_ddp, args):
    model_ema = None

    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(
            model_without_ddp, device=device, decay=1.0 - alpha
        )

    return model_ema


def get_data(args):
    if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        train_dir = None
        val_dir = None
        test_dir = None
    else:
        if args.dataset == "HAM10000":
            path = os.path.join(args.dataset_basepath, "HAM10000_dataset/")
        elif args.dataset == "fitzpatrick":
            path = os.path.join(args.dataset_basepath, "Fitzpatrick/")
        elif args.dataset == "breastUS":
            path = os.path.join(args.dataset_basepath, "Breast_US_dataset_split/")
        elif args.dataset == "retinopathy":
            path = os.path.join(args.dataset_basepath, "Retinopathy/")
        elif args.dataset == "pneumonia":
            path = os.path.join(args.dataset_basepath, "Pneumonia_Detection/")
        elif args.dataset == "smdg":
            path = os.path.join(args.dataset_basepath, "SMDG/")
        elif args.dataset == "papila":
            path = os.path.join(args.dataset_basepath, "PAPILA/")
        else:
            raise NotImplementedError

        train_dir = os.path.join(path, "train")
        val_dir = os.path.join(path, "val")
        test_dir = os.path.join(path, "test")

    # dataset, dataset_val, train_sampler, val_sampler = load_data(train_dir, val_dir, test_dir, args)

    (
        dataset,
        dataset_val,
        dataset_test,
        train_sampler,
        val_sampler,
        test_sampler,
    ) = load_data(train_dir, val_dir, test_dir, args)

    return dataset, dataset_val, dataset_test, train_sampler, val_sampler, test_sampler


def get_fairness_data(args, yaml_data):
    train_csv_path = yaml_data["data"][args.dataset]["train_csv"]
    val_csv_path = yaml_data["data"][args.dataset]["val_csv"]
    test_csv_path = yaml_data["data"][args.dataset]["test_csv"]
    img_path = yaml_data["data"][args.dataset]["img_path"]

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    if(args.dataset != "ol3i"):
        train_df["Path"] = train_df["Path"].apply(lambda x: os.path.join(img_path, x))
        val_df["Path"] = val_df["Path"].apply(lambda x: os.path.join(img_path, x))
        test_df["Path"] = test_df["Path"].apply(lambda x: os.path.join(img_path, x))
    else:
        pass

    (
        dataset,
        dataset_val,
        dataset_test,
        train_sampler,
        val_sampler,
        test_sampler,
    ) = load_fairness_data(args, train_df, val_df, test_df)

    return dataset, dataset_val, dataset_test, train_sampler, val_sampler, test_sampler
