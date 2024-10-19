import torch
from torch.utils.data import DataLoader

from datasets.pixel_imagenet import PixelImageNet
from helpers.linear_decompose import *
from helpers.inspect_utils import *
from helpers.utils import *
from helpers.model_utils import *
from helpers.decompose_utils import *
from helpers.interpret_utils import *

from sklearn.metrics import average_precision_score
import os
import argparse


args = argparse.ArgumentParser()
args.add_argument("--model_key", type=str, default="DINO")
args.add_argument("--num_samples", type=int, default=5000)
args.add_argument("--save_path", type=str, default='./saved_plots/zs_segementation_decompose.csv')
args = args.parse_args()

model_key = args.model_key
num_samples = args.num_samples
save_path = args.save_path


with open("./imagenet_classes.txt", "r") as fp:
    classes = [x.strip() for x in fp.readlines()]

with open("./templates.txt", "r") as fp:
    templates = [x.strip() for x in fp.readlines()]


num_workers = 4 * torch.cuda.device_count()
gpu_size = 512 * torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pht = "clip_zeroshot" if model_key == "CLIP" else "imgnet_trained"
model, model_descr, batch_size, pred_head = load_model(model_key, device, classes, templates, pred_head_type=pht)
if model_key == "SWIN":
    detach_block, end_block = (2, 10), (3,2)
elif model_key == "MaxVit":
    detach_block, end_block = (2,0), (3,2)
else:
    detach_block, end_block = 5, 12

model.detach_from_res(detach_block, end_block)
model.freeze_blocks(0, detach_block)

model.to(device)
_ = model.eval()


pixel_imagenet = PixelImageNet('/fs/cml-datasets/ImageNet/ILSVRC2012/', 
                             '/cmlscratch/sriramb/pixel-imagenet-v2/',
                             img_transform=model.preprocess, 
                             mask_transform=model.preprocess)
pixel_imagenet =  torch.utils.data.Subset(pixel_imagenet, torch.load('./pixel_imagenet_shuffled_indices.pkl')[:num_samples])
pixel_imagenet_loader = DataLoader(pixel_imagenet, batch_size=1, shuffle=False, num_workers=num_workers)


def segment_image(model, model_key, pred_head, image_batch, labels):
    model.expand_at_points(heads=False, tokens=True)
    
    probe_vec = pred_head(model(image_batch.to(device)))[torch.arange(len(image_batch)), labels]
    
    with torch.no_grad():
        embeds_decomp, _ = decompose(probe_vec.grad_fn, probe_vec, probe_vec.shape, 0,  
                                        Metadata(probe_vec.device, probe_vec.dtype))
        embeds_decomp = remove_singleton(embeds_decomp)
    model.collect_components(embeds_decomp)
    attn_maps = [model.get_attn_component(i) for i in range(len(model.attn_comps))]
    

    if model_key == 'SWIN':
        new_attn_maps = []
        for am in attn_maps:
            if len(am.shape) == 2:
                new_attn_maps.append(am.view(7, 7, *am.shape[1:]))
            else:
                new_attn_maps.append(am.view(7, 7, *am.shape[1:]).permute(2,0,3,1,4).reshape(14, 14, *am.shape[-1:]))
        attn_maps = new_attn_maps
    elif model_key == 'MaxVit':
        attn_maps = [am.reshape(7, 7, -1,) for am in attn_maps]
    else:
        attn_maps = [am[1:].reshape(14, 14, -1, *am.shape[2:]) for am in attn_maps]
    
    attn_maps = [T.functional.resize(am.permute(2,0,1), 224) for am in attn_maps]
    attn_maps = torch.stack(attn_maps).sum(dim=0)
    attn_maps = attn_maps[:,None]
    return attn_maps

total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
total_ap = []

predictions, targets = [], []

for i, (imgs, masks, labels) in enumerate(itr := tqdm(pixel_imagenet_loader)):
    masks = (masks > 0).int()
    masks =  masks[:,0:1]
    
    salmap = segment_image(model, model_key, pred_head, imgs, labels) # [b, 1, h, w]
    salmap = torch.clip(salmap, min=0)/ salmap.max()
    segmap = salmap.gt(salmap.mean()).int()


    correct = (segmap == masks).int().sum()
    labeled = len(masks.reshape(-1))
    
    inter = ((segmap == 1) & (masks == 1)).int().sum()
    union = ((segmap == 1) | (masks == 1)).int().sum()

    ap = average_precision_score(masks.reshape(-1), salmap.view(-1))

    total_correct += correct.to(torch.int64)
    total_label += torch.tensor(labeled).to(torch.int64)
    total_inter += inter.to(torch.int64)
    total_union += union.to(torch.int64)
    total_ap += [ap]
    
    pixAcc = total_correct / total_label
    mIoU = total_inter /  total_union

    mAp = np.mean(total_ap)
    itr.set_description('pixAcc: %.4f, mIoU: %.4f, mAP: %.4f' % (pixAcc, mIoU, mAp))

if not os.path.exists(save_path):
        # Write header
    with open(save_path, 'w') as fp:
        fp.write('model_key, pixAcc, mIoU, mAP\n')
    
with open(save_path, 'a') as fp:
    fp.write(f"{model_key},")
    fp.write(f"{np.round(pixAcc, 4)}, {np.round(mIoU, 4)}, {np.round(mAp, 4)}")
    fp.write('\n')
    