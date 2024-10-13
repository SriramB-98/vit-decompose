import torch
import numpy as np
from helpers.linear_decompose import *
from tqdm.auto import tqdm

def dict_to_tensor(tensor_dict):
    tensor_list = [t for _, t in sorted(tensor_dict.items(), key=lambda x: x[0])]
    return torch.stack(tensor_list, dim=0)
    
def concat_dict(tensor_dict_list):
    keys = sorted(tensor_dict_list[0].keys())
    tensor_dict = {k: torch.cat([td[k] for td in tensor_dict_list], dim=0) for k in keys}
    return tensor_dict

def head_outputs(model, layer_head_list=None, mlp_list=None, tensor_output=False):
    def new_model(inp):
        with torch.enable_grad():
            embeds = model(inp)
        with torch.no_grad():
            embeds_decomp, _ = decompose(embeds.grad_fn, embeds, embeds.shape, 0,  
                                               Metadata(embeds.device, embeds.dtype))
            embeds_decomp = remove_singleton(embeds_decomp)
        model.collect_components(embeds_decomp)
        
        comp_dict = dict()
        
        if model.expand_dict['heads']:
            all_layer_heads = [(li, hi) for li in range(len(model.attn_comps)) for hi in range(len(model.attn_comps[li]))]
        else:
            all_layer_heads = [(li, None) for li in range(len(model.attn_comps))]

        for layer_i, head_i in (layer_head_list or all_layer_heads):
            if head_i is None:
                comp_dict[f'layer:{layer_i:02}, attns'] = model.get_attn_component(layer_i)
            else:
                comp_dict[f'layer:{layer_i:02}, attn_head:{head_i:02}'] = model.get_attn_component(layer_i, head_i)
        
        for mlp_i in (mlp_list or range(len(model.mlp_comps))):
            comp_dict[f'layer:{mlp_i:02}, mlp'] = model.get_mlp_component(mlp_i)
        
        if getattr(model, 'conv_comps', None) is not None:
            for conv_i in range(len(model.conv_comps)):
                comp_dict[f'layer:({2*conv_i:02}, {2*conv_i+1:02}), conv'] = model.get_conv_component(conv_i)

        comp_dict['init'] = model.init
        
        if tensor_output:
            return dict_to_tensor(comp_dict).to(embeds.device)
        else:
            return comp_dict
    
    return new_model

def get_decomposed_embeds(model, dataloader, num_batches, device, layer_head_list=None, mlp_list=None, heads=True, tokens=False, 
                          load_file=None):
    
    try:
        if load_file is not None:
            comp_names, embeds, labels = torch.load(load_file, map_location='cpu')
            return comp_names, embeds, labels
        else:
            print('Generating embeddings from scratch!')
    except FileNotFoundError:
        print('File not found, generating embeddings from scratch!')
        
    model.expand_at_points(heads=heads, tokens=tokens)
    decomposed_model = head_outputs(model, layer_head_list=layer_head_list, mlp_list=mlp_list, tensor_output=False)

    embeds_list = []
    labels_list = []

    for i, batch in tqdm(enumerate(dataloader), total=num_batches):
        if i == num_batches:
            break
        imgs, labels = batch[0], torch.stack(batch[1:], dim=1)
        with torch.no_grad():
            embeds_dict = decomposed_model(imgs.to(device))
            embeds = dict_to_tensor(embeds_dict)
            if i == 0:
                comp_names = list(sorted(embeds_dict.keys()))
            embeds_list.append(embeds.cpu())
            labels_list.append(labels)

    embeds = torch.cat(embeds_list, dim=1)
    labels = torch.cat(labels_list, dim=0)
    
    if load_file is not None:
        torch.save((comp_names, embeds, labels), load_file)
    
    return comp_names, embeds, labels

# def headwise_contribution(embeds_decomp, bases, method='cov'):
#     bases_embeds_decomp = embeds_decomp@bases.T
#     if method == 'cov':
#         contributions = torch.Tensor([torch.cov(pe.t()).view(-1).sum().item() 
#                                       for pe in bases_embeds_decomp.unbind(0)])
#     elif method == 'ind_var':
#         contributions = bases_embeds_decomp.std(dim=1).mean(dim=-1)
#     contributions = contributions/contributions.sum()
#     return contributions

def trace(x):
    try:
        return torch.trace(x)
    except:
        return x

def variance_explained(embeds_decomp, bases):
    embeds_decomp = embeds_decomp - embeds_decomp.mean(dim=1, keepdims=True)
    bases = torch.qr(bases.T).Q.T
    bases_embeds_decomp = embeds_decomp @ bases.T  # @bases
    variations = torch.Tensor(
        [
            trace(torch.cov(bec.t())).item() / trace(torch.cov(ec.t())).item()
            for bec, ec in zip(bases_embeds_decomp.unbind(0), embeds_decomp.unbind(0))
        ]
    )
    return variations

def variance_attributed(embeds_decomp, bases, normalize=True):
    # embeds_decomp: num_comps x bs x d
    # bases: k x d
    bases = torch.qr(bases.T).Q.T
    total_scores = embeds_decomp.sum(0)@bases.T
    total_scores = total_scores - total_scores.mean(0) # shape: bs x k
    total_var = (total_scores**2).mean()
    
    # embeds_decomp = embeds_decomp - embeds_decomp.mean(dim=1, keepdims=True)
    embeds_score = embeds_decomp@bases.T 
    embeds_score = embeds_score - embeds_score.mean(dim=1, keepdims=True) #shape: num_comps x bs x k
    covar = (total_scores[None,:]*embeds_score).mean(dim=1)
    if normalize:
        covar = covar/(total_scores[None,:].std(dim=1)*embeds_score.std(dim=1))
    return covar.mean(-1)