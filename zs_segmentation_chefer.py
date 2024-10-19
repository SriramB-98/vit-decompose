import pickle

import torch
import timm
import matplotlib.pyplot as plt
from datasets.cached_imgnet import *
from captum.attr import visualization as viz

from torch.utils.data import DataLoader

from datasets.datamodules import *
from linear_decompose import *
from inspect_utils import *
from utils import *
from model_utils import *
from decompose_utils import *
from interpret_utils import *

from sklearn.metrics import f1_score, average_precision_score
import argparse

from transformer_explainability.baselines.ViT.ViT_LRP import *
from transformer_explainability.baselines.ViT.ViT_explanation_generator import Baselines, LRP

import cv2


args = argparse.ArgumentParser()
args.add_argument("--model_key", type=str, default="DINO")
args.add_argument("--num_samples", type=int, default=5000)
args.add_argument("--save_path", type=str, default='./saved_plots/zs_segementation_chefer.csv')
args.add_argument("--method", type=str, default="transformer_attribution")
args = args.parse_args()

model_key = args.model_key
num_samples = args.num_samples
save_path = args.save_path
method =args.method

with open("./imagenet_classes.txt", "r") as fp:
    classes = [x.strip() for x in fp.readlines()]

with open("./templates.txt", "r") as fp:
    templates = [x.strip() for x in fp.readlines()]


num_workers = 4 * torch.cuda.device_count()
gpu_size = 512 * torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# initialize ViT pretrained
if model_key == "DeiT":
    model = deit_base_patch16_224(pretrained=True).cuda()
elif model_key == "DINO":
    model = vit_base_patch16_224()
    dino_model, _, _, pred_head = load_model(model_key, device, classes=classes, templates=templates, pred_head_type="imgnet_trained")
    state_dict = dino_model.model.state_dict()
    state_dict['head.weight'] = pred_head.weight
    state_dict['head.bias'] = pred_head.bias
    model.load_state_dict(state_dict)
    model = model.cuda()
else:
    raise NotImplementedError
model.eval()

pixel_imagenet = PixelImageNet('/fs/cml-datasets/ImageNet/ILSVRC2012/', 
                             '/cmlscratch/sriramb/pixel-imagenet-v2/',
                             img_transform=transform, 
                             mask_transform=transform)
pixel_imagenet =  torch.utils.data.Subset(pixel_imagenet, torch.load('./pixel_imagenet_shuffled_indices.pkl')[:num_samples])
pixel_imagenet_loader = DataLoader(pixel_imagenet, batch_size=1, shuffle=False, num_workers=num_workers)


def generate_visualization(model, imgs, class_index=None, use_thresholding=True):
    if method == "transformer_attribution": 
        attribution_generator = LRP(model)
        transformer_attribution = attribution_generator.generate_LRP(imgs.cuda(), method="transformer_attribution", index=class_index, start_layer=1).detach()
    elif method == 'gradcam':
        baselines = Baselines(model)
        transformer_attribution = baselines.generate_cam_attn(imgs.cuda()).detach()

    transformer_attribution = transformer_attribution.reshape(len(imgs), 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    salmap = transformer_attribution

    thresh = salmap.mean()

    segmap = (salmap > thresh).astype(np.uint8)
        
    return torch.Tensor(salmap), torch.Tensor(segmap).int()


total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
total_ap = []

predictions, targets = [], []

for i, (imgs, masks, labels) in enumerate(itr := tqdm(pixel_imagenet_loader)):
    masks = (masks > 0).int()
    masks =  masks[:,0:1]
    imgs = normalize(imgs)

    salmap, segmap = generate_visualization(model, imgs, class_index=labels[0]) # [b, 1, h, w]

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
    fp.write(f"{model_key}, ")
    fp.write(f"{np.round(pixAcc, 4)}, {np.round(mIoU, 4)}, {np.round(mAp, 4)}")
    fp.write('\n')
    