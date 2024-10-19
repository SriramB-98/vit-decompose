from torchvision.datasets import ImageNet
import torch

from torch.utils.data import DataLoader

from helpers.linear_decompose import *
from helpers.inspect_utils import *
from helpers.utils import *
from helpers.model_utils import *
from helpers.decompose_utils import *
from helpers.interpret_utils import *
from torchmetrics.regression import SpearmanCorrCoef

set_seed(0)


model_keys = ["DeiT", "DINO", "DINOv2", "SWIN", "MaxVit"] #
save_path = './saved_plots/img_retrieval_from_text.csv'
imgnet_path = None # replace with path to ImageNet dataset
if imgnet_path is None:
    raise ValueError("Please provide path to ImageNet dataset")

num_feats = 4

feat_desc_dict = {
    "color": ["blue color", "green color", "red color", "yellow color", "black color", "white color"],
    "texture": [
        "rough texture",
        "smooth texture",
        "furry texture",
        "sleek texture",
        "slimy texture",
        "spiky texture",
        "glossy texture",
    ],
    'animal': [ 'cat', 'dog', 'camel', 'elephant', 'giraffe', 'zebra', 'cheetah'],
    "person": ["face", "head", "man", "woman", "human", "arms", "legs"],
    "location": ["beach", "forest", "desert", "sea", "city", "sky", "marsh"],
    "pattern": ["spotted pattern", "striped pattern", "polka dot pattern", "plain pattern", "checkered pattern"],
    "shape": ["triangular shape", "rectangular shape", "circular shape", "octagon"],
}


with open("./imagenet_classes.txt", "r") as fp:
    classes = [x.strip() for x in fp.readlines()]

with open("./templates.txt", "r") as fp:
    templates = [x.strip() for x in fp.readlines()]


num_workers = 4 * torch.cuda.device_count()
gpu_size = 512 * torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


clip_type="openclip"
clip_model_name="ViT-L/14"
pretrained="laion2b_s32b_b82k"
clip_descr = f"{clip_type}_{clip_model_name}_{pretrained}".replace('/', '-')
clip_model = CLIPModel(clip_type, clip_model_name, pretrained).to(device).eval()

try:
    clip_embeds, labels = torch.load(f"./saved_outputs/{clip_descr}_imagenet_embeds.pt")
    feat_embeds_list = []
    for feat, feat_desc in feat_desc_dict.items():
        feat_embeds = get_clip_text_embeds(clip_model, feat_desc[:num_feats], templates, device).weight.data.cpu()
        feat_embeds_list.append(feat_embeds)
    feat_embeds = torch.stack(feat_embeds_list)
    clip_rankings = torch.einsum('ij,jkl->lki', clip_embeds, feat_embeds.T)
#     clip_rankings = clip_rankings - clip_rankings.mean(dim=-1, keepdims=True)
except:
    clip_dataset = ImageNet(imgnet_path, split="val", 
                            transform=clip_model.preprocess)

    clip_dataloader = DataLoader(clip_dataset, batch_size=128, shuffle=True, num_workers=num_workers)

    clip_embeds_list = []
    labels_list = []

    num_batches = 6400 // clip_dataloader.batch_size
    for i, batch in tqdm(enumerate(clip_dataloader), total=num_batches):
        if i == num_batches:
            break
        imgs, labels = batch[0], torch.stack(batch[1:], dim=1)
        with torch.no_grad():
            clip_img_features = clip_model(imgs.to(device))
            clip_embeds_list.append(clip_img_features.cpu())
            labels_list.append(labels)

    clip_embeds = torch.cat(clip_embeds_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    torch.save((clip_embeds, labels), f"./saved_outputs/{clip_descr}_imagenet_embeds.pt")


for model_key in model_keys:
    pht = "clip_zeroshot" if model_key == "CLIP" else "imgnet_trained"
    model, model_descr, batch_size, pred_head = load_model(model_key, device, classes, 
                                                           templates, pred_head_type=pht)
    if model_key == "SWIN":
        detach_block, end_block = (2, 14), (3,2)
    elif model_key == "MaxVit":
        detach_block, end_block = (2,3), (3,2)
    else:
        detach_block, end_block = 7, 12

    print(detach_block, end_block)
    model.detach_from_res(detach_block, end_block)
    model.freeze_blocks(0, detach_block)

    model.to(device)
    _ = model.eval()

    clip_model, clip_model_descr, clip_aligner_head = get_clip_and_aligner(model, model_descr, device)

    dataset = ImageNet(imgnet_path, split="val", transform=model.preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    comp_names, embeds_decomp, labels = get_decomposed_embeds(
        model,
        dataloader,
        6400 // dataloader.batch_size,
        device,
        load_file=f"./saved_outputs/{model_descr}_imgnet_decomposed_embeds.pt",
    )
    with torch.no_grad():
        clip_aligned_embeds_decomp = clip_aligner_head(embeds_decomp)    
    
#     feat_embeds_list = []
#     for feat, feat_desc in feat_desc_dict.items():
#         feat_embeds = get_clip_text_embeds(clip_model, feat_desc[:num_feats], templates, device).weight.data.cpu()
#         feat_embeds_list.append(feat_embeds)
#     feat_embeds = torch.stack(feat_embeds_list)
    
    with torch.no_grad():
        ranking_per_probe_comp = torch.einsum('hij,jkl->lkhi', clip_aligned_embeds_decomp, feat_embeds.T)

    ranking_per_probe_comp = ranking_per_probe_comp - ranking_per_probe_comp.mean(dim=-1, keepdims=True)
    order_corrs = (ranking_per_probe_comp*clip_rankings[:,:,None]).mean(-1)
    order_corrs_coeff = order_corrs/(ranking_per_probe_comp.std(dim=-1)*clip_rankings[:,:,None].std(dim=-1))

    feat_to_var_attr = torch.stack([variance_attributed(clip_aligned_embeds_decomp, fe_i) 
                                         for fe_i in feat_embeds])

    spearman_corr_coeff = SpearmanCorrCoef(num_outputs=len(feat_to_var_attr.T))
    within_comp_ordering = spearman_corr_coeff(order_corrs_coeff.mean(1), feat_to_var_attr).mean()
    
    spearman_corr_coeff = SpearmanCorrCoef(num_outputs=len(feat_to_var_attr))
    between_comp_ordering = spearman_corr_coeff(order_corrs_coeff.mean(1).T, feat_to_var_attr.T)
    
    if not os.path.exists(save_path):
        # Write header
        with open(save_path, 'w') as fp:
            fp.write('model_key, within_comp_ordering, between_comp_ordering (average),' + ','.join(feat_desc_dict.keys()))
            fp.write('\n')

    with open(save_path, 'a') as fp:
        rest_of_the_row = ", ".join([f"{x:.3f}" for x in between_comp_ordering])
        fp.write(f"{model_key}, {within_comp_ordering:.3f}, {between_comp_ordering.mean():.3f}, "+rest_of_the_row)
        fp.write('\n')

