import os
import numpy as np

import timm
import torch
import torchvision
import torchvision.transforms as transforms

import random

def disable_grad(module):
    for p in module.parameters():
        p.requires_grad = False


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

    return vector

def tune_attention_layers(model):
    
    for name_p,p in model.named_parameters():
        if '.attn.' in name_p or 'attention' in name_p:
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
                

def tune_layernorm_layers(model):

    disable_grad(model)

    for n,p in model.named_parameters():
        if("norm" in n or "head" in n):
            p.requires_grad = True

    return model

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
                x = random.randint(0,1)
                if(x >= 0.5):
                    param.requires_grad = True
                else:
                    continue
                break


model = timm.create_model("vit_base_patch16_224", pretrained=True)

disable_grad(model)
#tune_attention_layers(model)
print(tune_attention_layers_random(model))
check_tunable_params(model, False)


