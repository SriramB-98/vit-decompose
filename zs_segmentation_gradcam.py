import torch
import timm
from datasets.cached_imgnet import *

from pytorch_grad_cam import GradCAM
import timm
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from torch.utils.data import DataLoader

from datasets.datamodules import *
from linear_decompose import *
from inspect_utils import *
from utils import *
from model_utils import *
from decompose_utils import *
from interpret_utils import *

from sklearn.metrics import average_precision_score
import argparse

from transformer_explainability.baselines.ViT.ViT_LRP import *
from transformer_explainability.baselines.ViT.ViT_explanation_generator import Baselines, LRP

import cv2


args = argparse.ArgumentParser()
args.add_argument("--model_key", type=str, default="DINO")
args.add_argument("--num_samples", type=int, default=5000)
args.add_argument("--save_path", type=str, default='./saved_plots/zs_segementation_gradcam.csv')
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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

if model_key == 'MaxVit': 
    model = timm.create_model("maxvit_small_tf_224.in1k", pretrained=True)
    def reshape_transform(tensor, height=7, width=7):
        result = tensor.reshape(tensor.size(0),
            height, width, tensor.size(3))
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    target_layers = [model.stages[-1].blocks[-1].attn_block.norm1]
    
elif model_key == 'SWIN':
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)

    def reshape_transform(tensor, height=7, width=7):
        result = tensor.reshape(tensor.size(0),
            height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    
    target_layers = [model.layers[-1].blocks[-1].norm2]
    
else:
    if model_key == 'DeiT':
        model = timm.create_model("deit_base_patch16_224", pretrained=True)
    else:
        model = timm.create_model("vit_base_patch16_224.dino", pretrained=True)
        _,  _, _, pred_head = load_model(model_key, device, classes=classes, templates=templates, pred_head_type="imgnet_trained")
        model.head = pred_head

    def reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1 :  , :].reshape(tensor.size(0),
            height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    target_layers = [model.blocks[-1].norm1]
    

cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
model.to(device)
model.eval()

pixel_imagenet = PixelImageNet('/fs/cml-datasets/ImageNet/ILSVRC2012/', 
                             '/cmlscratch/sriramb/pixel-imagenet-v2/',
                             img_transform=transform, 
                             mask_transform=transform)
pixel_imagenet =  torch.utils.data.Subset(pixel_imagenet, torch.load('./pixel_imagenet_shuffled_indices.pkl')[:num_samples])
pixel_imagenet_loader = DataLoader(pixel_imagenet, batch_size=1, shuffle=False, num_workers=num_workers)


def generate_visualization(imgs, labels=None, use_thresholding=True):
    
    targets = [ClassifierOutputTarget(l.item()) for l in labels]
    grayscale_cam = cam(input_tensor=imgs, targets=targets)

    grayscale_cam = grayscale_cam.reshape(224, 224)
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)

    salmap = grayscale_cam
    # if use_thresholding:
    #     grayscale_cam = grayscale_cam * 255
    #     grayscale_cam = grayscale_cam.astype(np.uint8)
    #     ret, grayscale_cam = cv2.threshold(grayscale_cam, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     grayscale_cam[grayscale_cam == 255] = 1

    thresh = salmap.mean()

    segmap = (salmap > thresh).astype(np.uint8)


    return torch.Tensor(salmap), torch.Tensor(segmap).int()


total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
total_ap = []

predictions, targets = [], []

for i, (imgs, masks, labels) in enumerate(itr := tqdm(pixel_imagenet_loader)):
    masks = (masks > 0).int()
    masks =  masks[:,0:1]
    imgs = normalize(imgs.to(device))

    salmap, segmap = generate_visualization(imgs, labels=labels) # [b, 1, h, w]

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
    