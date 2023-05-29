import os
os.environ["TORCH_HOME"] = os.path.dirname(os.getcwd())

import copy
import datetime
import re
import errno
import hashlib
import numpy as np

import time
from collections import defaultdict, deque, OrderedDict
from typing import List, Optional, Tuple
import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

import matplotlib.pyplot as plt
import seaborn as sns


###################### HELPER FUNCTIONS #######################


def disable_module(module):
    for p in module.parameters():
        p.requires_grad = False
        
def enable_module(module):
    for p in module.parameters():
        p.requires_grad = True


def check_tunable_params(model, verbose=True):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            if(verbose):
                print(name)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.5f}"
    )

    return trainable_params, all_param

def enable_from_vector(vector, model):

    print("Vector: ", vector)
    
    disable_module(model)
    
    for idx, block in enumerate(model.blocks): 
    
        if(vector[idx] == 1):
            print("Enabling attention in Block {}".format(idx))
            enable_module(block.attn)
        else:
            #print("Disabling attention in Block {}".format(idx))
            disable_module(block.attn)


def tune_attention_layers_random(model, model_type='timm'):

    vector = []
    
    for name_p,p in model.named_parameters():
        if '.attn.' in name_p or 'attention' in name_p:
            if(np.random.random(1)[0] >= 0.5):
                vector.append(1)
                p.requires_grad = True
            else:
                vector.append(0)
        else:
            p.requires_grad = False
        try:
            #Timm Model
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            #HF Model
            model.classifier.weight.requires_grad = True
            model.classifier.bias.requires_grad = True
        
        # POSITION EMBEDDING
        if(model_type == 'timm'):
            try:
                model.pos_embed.requires_grad = True
            except:
                print('no pos embedding')
        elif(model_type == 'hf'):
            try:
                model.vit.embeddings.position_embeddings.requires_grad = True
            except:
                print('no pos embedding')
            
        # PATCH EMBEDDING
        if(model_type == 'timm'):
            try:
                for p in model.patch_embed.parameters():
                    p.requires_grad = False
            except:
                print('no patch embed')
                
        elif(model_type == 'hf'):
            try:
                for p in model.vit.embeddings.patch_embeddings.parameters():
                    p.requires_grad = False
            except:
                print('no patch embed')

    #print("MASKING VECTOR: ", vector)
    return vector


def tune_blocks_random(model, segment):
    vector = []

    for idx, block in enumerate(model.blocks):
        if(np.random.random(1)[0] >= 0.5):
            print("Enabling {} in Block {}".format(segment, idx))
            if(segment == 'attention'):
                enable_module(block.attn)
            elif(segment == 'layernorm'):
                enable_module(block.norm1)
                enable_module(block.norm2)

            vector.append(1)
        else:
            print("Disabling {} in Block {}".format(segment, idx))
            if(segment == 'attention'):
                disable_module(block.attn)
            elif(segment == 'layernorm'):
                disable_module(block.norm1)
                disable_module(block.norm2)
            
            vector.append(0)
            
    return vector

def tune_attention_layers(model):

    vector = []
    
    for name_p,p in model.named_parameters():
        if '.attn.' in name_p or 'attention' in name_p:
            vector.append(1)
            p.requires_grad = True
        else:
            p.requires_grad = False
        
        #Timm Model
        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True
    
        
        # POSITION EMBEDDING
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no pos embedding')
        
        # PATCH EMBEDDING
        
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')

    return vector
                

def tune_layernorm_layers(model):

    disable_module(model)

    vector = []

    for n,p in model.named_parameters():
        if("norm" in n or "head" in n):
            vector.append(1)
            p.requires_grad = True

    return vector

def tune_layernorm_random(model):

    disable_module(model)

    vector = []

    for n,p in model.named_parameters():
        if("norm" in n or "head" in n):
            if(np.random.random(1)[0] >= 0.5):
                vector.append(1)
                p.requires_grad = True
            else:
                vector.append(0)

    return vector


def get_model_for_bitfit(model, model_type):
    trainable_components = ['bias'] 

    # Disale all the gradients
    for param in model.parameters():
        param.requires_grad = False 
    
    #Add classification head to trainable components
    if trainable_components:
        trainable_components = trainable_components + ['pooler.dense.bias']
        
    if(model_type == 'timm'):
        trainable_components = trainable_components + ['head']
    elif(model_type == 'hf'):
        trainable_components = trainable_components + ['classifier']

    vector = []

    for name, param in model.named_parameters():
        for component in trainable_components:
            if component in name:
                vector.append(1)
                param.requires_grad = True
                break
    
    return vector

def get_model_bitfit_random(model):
    trainable_components = ['bias'] 

    # Disale all the gradients
    for param in model.parameters():
        param.requires_grad = False 
    
    #Add classification head to trainable components
    if trainable_components:
        trainable_components = trainable_components + ['pooler.dense.bias']

    trainable_components = trainable_components + ['head']

    vector = []

    for name, param in model.named_parameters():
        for component in trainable_components:
            if component in name:
                if(np.random.random(1)[0] >= 0.5):
                    vector.append(1)
                    param.requires_grad = True
                else:
                    vector.append(0)

    return vector

def get_timm_model(encoder, num_classes, **kwargs):
    '''
    Returns a timm model for a given encoder.
    '''

    assert num_classes is not None, "Number of classes cannot be None"
    if encoder == "vit_base":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes) 
    elif encoder == "vit_large":
        model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=num_classes)
    elif encoder == "vit_huge":
        model = timm.create_model("vit_huge_patch14_224", pretrained=True, num_classes=num_classes)

    return model

def get_masked_model(model, method):
    if(method == 'fullft'):
        pass
    if(method == 'tune_attention'):
        disable_module(model)
        vector = tune_attention_layers(model)
    elif(method == 'tune_attention_random'):
        disable_module(model)
        vector = tune_attention_layers_random(model)
    elif(method == 'tune_attention_blocks_random'):
        disable_module(model)
        vector = tune_blocks_random(model, segment='attention')
    elif(method == 'bitfit'):
        vector = get_model_for_bitfit(model, 'timm')
    elif(method == 'tune_bitfit_random'):
        vector = get_model_bitfit_random(model)
    elif(method == 'tune_layernorm'):
        disable_module(model)
        vector = tune_layernorm_layers(model)
    elif(method == 'tune_layernorm_random'):
        disable_module(model)
        vector = tune_layernorm_random(model)
    elif(method == 'tune_layernorm_blocks_random'):
        disable_module(model)
        vector = tune_blocks_random(model, segment='layernorm')
    else:
        raise NotImplementedError

    enable_module(model.head)

    return vector

def get_model_from_vector(model, method, vector):
    if(method == 'tune_attention_blocks_random'):
        disable_module(model)
        enable_from_vector(vector, model)


def plot_changes(fine_tuned_ckpt, base_model, args):
    '''
    Plots the changes in different layers of a model
    '''

    if(args.model == 'vit_base'):
        num_layers = 12
    if(args.model == 'vit_large'):
        num_layers = 24
    if(args.model == 'vit_huge'):
        num_layers = 32

    fine_tuned_model = get_timm_model(args.model, args.num_classes)
    ckpt = torch.load(fine_tuned_ckpt)
    fine_tuned_model.load_state_dict(ckpt["model"], strict=True)
    fine_tuned_model = fine_tuned_model.cpu()

    def _calc_mean_diff(ft_p, base_p):
        return np.mean(np.abs(np.array(ft_p.data - base_p.data)))

    def _get_component_name(name):
        return re.split(r'.[0-9]+.', name)[1]

    def _get_component_layer(name):
        return int(name.split('.')[1])

    base_model = base_model.cpu()
    #fine_tuned_model = base_model.cpu()
    print(fine_tuned_ckpt)

    changes = []
    for ft_name, ft_param in fine_tuned_model.named_parameters():
        if ft_param.requires_grad and ('.attn.' in ft_name or 'attention' in ft_name):
            for base_name, base_param in base_model.named_parameters():
                if ft_name == base_name:
                    changes.append({'name': ft_name, 'value': _calc_mean_diff(ft_param, base_param)})

    keys = list(set(_get_component_name(c['name']) for c in changes))
    keys_mapper = {k: i for i, k in enumerate(keys)}

    total_weights = np.zeros(len(keys))
    for change in changes:
        total_weights[keys_mapper[_get_component_name(change['name'])]] += change['value']

    keys = [keys[i] for i in np.argsort(-total_weights)]
    keys_mapper = {k: i for i, k in enumerate(keys)}

    avg_column = np.zeros(len(keys))
    values_map = np.zeros((len(keys), num_layers + 1))
    for change in changes:
        avg_column[keys_mapper[_get_component_name(change['name'])]] += change['value']
        values_map[keys_mapper[_get_component_name(change['name'])], _get_component_layer(change['name'])] = change[
            'value']
    avg_column /= num_layers
    values_map[:, -1] = avg_column

    print(values_map)

    fig, ax = plt.subplots(figsize=(num_layers, len(keys)))
    xticklabels = [f'Layer {i}' for i in range(num_layers)]
    xticklabels.append('Avg.')
    yticklabels = keys
    sns.heatmap(values_map, cmap="Blues", ax=ax, xticklabels=xticklabels, yticklabels=yticklabels)

    filename = args.dataset + '_' + args.tuning_method + '_' + str(vector_idx)
    plt.savefig(os.path.join(args.fig_savepath, filename + '.png'))

# Calibration error scores in the form of loss metrics
class ECELoss(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    #     args.rank = int(os.environ["RANK"])
    #     args.world_size = int(os.environ["WORLD_SIZE"])
    #     args.gpu = int(os.environ["LOCAL_RANK"])
    # elif "SLURM_PROCID" in os.environ:
    #     args.rank = int(os.environ["SLURM_PROCID"])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # elif hasattr(args, "rank"):
    #     pass
    # else:
    #     print("Not using distributed mode")
    #     args.distributed = False
    #     return

    # args.distributed = True

    # torch.cuda.set_device(args.gpu)
    # args.dist_backend = "nccl"
    # print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    # torch.distributed.init_process_group(
    #     backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    # )
    # torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)

    args.distributed = False
    return


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16

    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(lambda s, _: torch.serialization.default_restore_location(s, "cpu")),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                f"For checkpoint {f}, expected list of params: {params_keys}, but found: {model_params_keys}"
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def store_model_weights(model, checkpoint_path, checkpoint_key="model", strict=True):
    """
    This method can be used to prepare weights files for new models. It receives as
    input a model architecture and a checkpoint from the training script and produces
    a file with the weights ready for release.

    Examples:
        from torchvision import models as M

        # Classification
        model = M.mobilenet_v3_large(weights=None)
        print(store_model_weights(model, './class.pth'))

        # Quantized Classification
        model = M.quantization.mobilenet_v3_large(weights=None, quantize=False)
        model.fuse_model(is_qat=True)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        _ = torch.ao.quantization.prepare_qat(model, inplace=True)
        print(store_model_weights(model, './qat.pth'))

        # Object Detection
        model = M.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
        print(store_model_weights(model, './obj.pth'))

        # Segmentation
        model = M.segmentation.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True)
        print(store_model_weights(model, './segm.pth', strict=False))

    Args:
        model (pytorch.nn.Module): The model on which the weights will be loaded for validation purposes.
        checkpoint_path (str): The path of the checkpoint we will load.
        checkpoint_key (str, optional): The key of the checkpoint where the model weights are stored.
            Default: "model".
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        output_path (str): The location where the weights are saved.
    """
    # Store the new model next to the checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Deep copy to avoid side effects on the model object.
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load the weights to the model to validate that everything works
    # and remove unnecessary weights (such as auxiliaries, etc.)
    if checkpoint_key == "model_ema":
        del checkpoint[checkpoint_key]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint[checkpoint_key], "module.")
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups
