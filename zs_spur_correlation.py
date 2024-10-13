from itertools import chain
import torch
from torchvision import transforms as T

import numpy as np
from torch.utils.data import DataLoader

from datasets.clean_waterbirds import *
from helpers.linear_decompose import *
from helpers.inspect_utils import *
from helpers.utils import *
from helpers.model_utils import *
from helpers.decompose_utils import *
from helpers.interpret_utils import *

set_seed(0)

model_keys = ["DeiT", "CLIP", "DINO", "DINOv2", "SWIN", "MaxVit"]
dataset_name = 'waterbirds'
correlated_waterbirds_dataset = None # replace with path to correlated dataset
uncorrelated_waterbirds_dataset = None # replace with path to uncorrelated dataset
num_comps = 10

save_path = f'./saved_plots/zs_spur_correlation_{dataset_name}.csv'

def get_spur_inds(embeds_decomp, spur_embeds, core_embeds, temp=1.0,  method='var'):
        
    with torch.no_grad():
        spur_vars = variance_attributed(clip_aligner_head(embeds_decomp), spur_embeds)
        core_vars = variance_attributed(clip_aligner_head(embeds_decomp), core_embeds)
    
    spur_to_core_ratio = spur_vars - core_vars

    sorted_inds = torch.argsort(spur_to_core_ratio, descending=True)
    
    return sorted_inds, spur_to_core_ratio

def mitigate_spur(embeds_decomp, sorted_inds, topk=10):
    spur_inds = sorted_inds[:topk]
    abl_embeds_decomp = embeds_decomp.clone()
    abl_embeds_decomp[spur_inds] = abl_embeds_decomp[spur_inds].mean(dim=1, keepdims=True)
    return abl_embeds_decomp

def get_group_accuracies(pred_head, embeds, labels, group_dict):
    acc_dict = dict()
    for grp_desc, grp_labels in group_dict.items():
        embeds_grp = embeds[grp_labels]
        labels_grp = labels[grp_labels]
        preds = pred_head(embeds_grp)
        acc_dict[grp_desc] = (preds.argmax(dim=-1) == labels_grp).float().mean().item()
    return acc_dict


with open("./templates.txt", "r") as fp:
    templates = [x.strip() for x in fp.readlines()]

num_workers = 4*torch.cuda.device_count()
gpu_size = 512*torch.cuda.device_count()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for model_key in model_keys:

    model, model_descr, batch_size, pred_head = load_model(model_key, device, pred_head_type=None)
    if model_key == "SWIN":
        detach_block, end_block = (2, 14), (3,2)
    elif model_key == "MaxVit":
        detach_block, end_block = (2,3), (3,2)
    else:
        detach_block, end_block = 7, 12
    model.detach_from_res(detach_block, end_block)
    model.freeze_blocks(0, detach_block)

    model.to(device)
    _=model.eval()

    waterbirds_classes =  ['landbird', 'waterbird']
    classes = waterbirds_classes
    
    super_dataset = CleanWaterbirdsDataset(correlated_waterbirds_dataset, transform=model.preprocess)
    train_dataset, val_dataset = torch.utils.data.random_split(super_dataset, [0.8, 0.2])
    test_dataset = CleanWaterbirdsDataset(uncorrelated_waterbirds_dataset, transform=model.preprocess)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=num_workers)

    
    if 'clip' in model_descr:
        pred_head = get_clip_text_embeds(model, waterbirds_classes, templates, device, 
                        load_file=f"./saved_outputs/{model_descr}_wb_zeroshot_classes_head.pt")
    else:
        pred_head = get_head(model, train_dataloader, len(classes), device, bias=True, 
                            load_file=f"./saved_outputs/{model_descr}_wb_trained_head.pt", 
                            epochs=10)
    pred_head.cpu()
        
    comp_names, embeds_decomp, labels = \
            get_decomposed_embeds(model, chain(val_dataloader, test_dataloader),
                                len(val_dataloader) + len(test_dataloader), device,
                                load_file=f'./saved_outputs/{model_descr}_waterbirds_decomposed_embeds.pt')
    labels, bg_labels = labels[:,0], labels[:,1]
    
    spur_feats_desc = ['water', 'ocean', 'lake', 'sky', 'land', 'forest', 'trees', 'bamboo', 'leaves']
    core_feats_desc =  ['bird', 'finch', 'hummingbird', 'crow', 'sparrow', 'swan', 'albatross', 'pelican']
    
    group_dict = {
        'waterbird on water': (labels == 1) & (bg_labels == 1),
        'waterbird on land': (labels == 1) & (bg_labels == 0),
        'landbird on water': (labels == 0) & (bg_labels == 1),
        'landbird on land': (labels == 0) & (bg_labels == 0)
    }

    clip_model, clip_model_descr, clip_aligner_head = get_clip_and_aligner(model, model_descr, device)

    spur_feat_embeds = get_clip_text_embeds(clip_model, spur_feats_desc, templates, device).weight.data.cpu()
    core_feat_embeds = get_clip_text_embeds(clip_model, core_feats_desc, templates, device).weight.data.cpu()

    orig_grp_acc_dict = get_group_accuracies(pred_head, embeds_decomp.sum(0), labels, group_dict)
    sorted_inds, _ = get_spur_inds(embeds_decomp, spur_feat_embeds, core_feat_embeds, temp=1, method='var')
    new_embeds_decomp = mitigate_spur(embeds_decomp, sorted_inds, topk=num_comps)
    new_grp_acc_dict = get_group_accuracies(pred_head, new_embeds_decomp.sum(0), labels, group_dict)

    if not os.path.exists(save_path):
        # Write header
        with open(save_path, 'w') as fp:
            fp.write('model_key,num_comps')
            for k in orig_grp_acc_dict.keys():
                fp.write(f",{k}")
            fp.write(f", worst group accuracy, average group accuracy")
            fp.write('\n')

    with open(save_path, 'a') as fp:
        fp.write(f"{model_key},{num_comps}")
        for k, v in orig_grp_acc_dict.items():
            fp.write(f", {v:.3f} $\\rightarrow$ {new_grp_acc_dict[k]:.3f}")
        fp.write(f", {min(orig_grp_acc_dict.values()):.3f}  $\\rightarrow$ {min(new_grp_acc_dict.values()):.3f}," 
                 f"{np.mean(list(orig_grp_acc_dict.values())):.3f}  $\\rightarrow$ {np.mean(list(new_grp_acc_dict.values())):.3f}")
        fp.write('\n')
