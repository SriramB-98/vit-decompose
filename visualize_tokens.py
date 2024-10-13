import pickle

from torchvision.datasets import ImageNet
import torch
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import visualization as viz

from torch.utils.data import DataLoader

from helpers.linear_decompose import *
from helpers.inspect_utils import *
from helpers.utils import *
from helpers.model_utils import *
from helpers.decompose_utils import *
from helpers.interpret_utils import *


model_keys = [
  "DeiT",
  "CLIP",
  "DINO",
  "DINOv2",
  "SWIN",
  "MaxVit"
 ]

imgnet_path = None # replace with path to ImageNet dataset

feat_list = [ "person", "pattern", "location" ]
feat_inst_list = ["man", 'striped pattern', 'beach']
probe_img_name = 'striped_shirt_beach'

# feat_list = ["animal", "location", "shape"]
# feat_inst_list = ["camel", "desert", "triangle"]
# probe_img_name = "camel_in_egypt"

num_rows = 3
num_comps = 3
num_batches = 100
visualize = True


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
    'animal': ['camel', 'elephant', 'giraffe', 'lion', 'tiger', 'zebra', 'cheetah'],
    "person": ["face", "head", "man", "woman", "human", "arms", "legs"],
    "location": ["sea", "beach", "forest", "desert", "city", "sky", "marsh"],
    "pattern": ["spotted pattern", "striped pattern", "polka dot pattern", "plain pattern", "checkered pattern"],
    "shape": ["triangular shape", "rectangular shape", "circular shape", "octagon"],
    "vehicle": ["car", "truck", "jeep", "van"],
}


with open("./imagenet_classes.txt", "r") as fp:
    classes = [x.strip() for x in fp.readlines()]

with open("./templates.txt", "r") as fp:
    templates = [x.strip() for x in fp.readlines()]


num_workers = 4 * torch.cuda.device_count()
gpu_size = 512 * torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_key in model_keys:
    pht = "clip_zeroshot" if model_key == "CLIP" else "imgnet_trained"
    model, model_descr, batch_size, pred_head = load_model(model_key, device, classes, templates, pred_head_type=pht)
    if model_key == "SWIN":
        detach_block, end_block = (2, 14), (3, 2)
    elif model_key == "MaxVit":
        detach_block, end_block = (2, 3), (3, 2)
    else:
        detach_block, end_block = 7, 12

    model.detach_from_res(detach_block, end_block)
    model.freeze_blocks(0, detach_block)

    model.to(device)
    _ = model.eval()

    clip_model, clip_model_descr, clip_aligner_head = get_clip_and_aligner(model, model_descr, device)

    dataset = ImageNet(imgnet_path, split="val", transform=model.preprocess)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    comp_names, embeds_decomp, labels = get_decomposed_embeds(
        model,
        dataloader,
        num_batches,
        device,
        load_file=f"./saved_outputs/{model_descr}_imgnet_decomposed_embeds.pt",
    )
    with torch.no_grad():
        clip_aligned_embeds_decomp = clip_aligner_head(embeds_decomp)

    attn_inds = torch.Tensor(["attn" in name for name in comp_names]).bool()
    clip_aligned_embeds_decomp = clip_aligned_embeds_decomp[attn_inds]
    if model_key != 'CLIP':
        clip_aligner_head_weights = clip_aligner_head.weights.data[attn_inds]
    comp_names = np.array(comp_names)[attn_inds]

    probe_img = model.preprocess(Image.open(f"./probe_imgs/{probe_img_name}.jpg"))

    model.expand_at_points(heads=True, tokens=True)
    probe_vec = model(probe_img[None, :].to(device))
    with torch.no_grad():
        probe_vec_decomp, _ = decompose(
            probe_vec.grad_fn, probe_vec, probe_vec.shape, 0, Metadata(probe_vec.device, probe_vec.dtype)
        )
        probe_vec_decomp = remove_singleton(probe_vec_decomp)
    model.collect_components(probe_vec_decomp)

    heat_map_list = []
    head_list = []

    sorted_heads_list = []
    variations_list = []
    for feat in feat_list:
        feat_desc = feat_desc_dict[feat]
        feat_embeds = get_clip_text_embeds(clip_model, feat_desc, templates, device).weight.data.cpu()
        variations_list.append(variance_explained(clip_aligned_embeds_decomp, feat_embeds))
    variations = torch.stack(variations_list) 
        
    sorted_heads_list = []
    for i in range(len(variations)):
        variations_2 = variations.clone()
        variations_2[i] = variations_2[i]*0
        sorted_heads_list.append(torch.argsort(variations[i] - 0*variations_2.max(dim=0).values, descending=True))
        
    for feat_inst, sorted_heads in zip(feat_inst_list, sorted_heads_list):
        rel_heads = comp_names[sorted_heads[:num_comps]]
        if num_comps == 1:
            rel_heads = [rel_heads]
        head_list.append(rel_heads)
        rel_attn_heads = [(int(hn[6:8]), int(hn[20:])) for hn in rel_heads]
        probe_text = get_clip_text_embeds(clip_model, [feat_inst], templates, device).weight.data.cpu()[0]

        if model_key == 'CLIP':
            tokens = [model.get_attn_component(l, h) for l, h in rel_attn_heads]
            tokens = [tok[1:].reshape(14, 14, *tok.shape[1:]) for tok in tokens]
        elif model_key == 'SWIN':
            tokens = [
                model.get_attn_component(l, h) @ clip_aligner_head_weights[i]
                for i, (l, h) in zip(sorted_heads[:num_comps], rel_attn_heads)
            ]
            new_tokens = []
            for t in tokens:
                if len(t.shape) == 3:
                    new_tokens.append(t.view(7, 7, *t.shape[1:]))
                else:
                    new_tokens.append(t.view(7, 7, *t.shape[1:]).permute(2,0,3,1,4,5).reshape(14, 14, *t.shape[-2:]))
            tokens = new_tokens   
        elif model_key == 'MaxVit':
            tokens = [
                model.get_attn_component(l, h) @ clip_aligner_head_weights[i]
                for i, (l, h) in zip(sorted_heads[:num_comps], rel_attn_heads)
            ]
            tokens = [tok.reshape(7, 7, *tok.shape[1:]) for tok in tokens]
        else:
            tokens = [
                model.get_attn_component(l, h) @ clip_aligner_head_weights[i]
                for i, (l, h) in zip(sorted_heads[:num_comps], rel_attn_heads)
            ]
            tokens = [tok[1:].reshape(14, 14, *tok.shape[1:]) for tok in tokens]
        
        heat_maps = [T.functional.resize((tok @ probe_text).squeeze()[None, :], 224) for tok in tokens]
        heat_map_list.append(torch.stack(heat_maps).sum(0))

        # heat_map = torch.stack([(tok @ probe_text).squeeze() for tok in tokens]).sum(0)

    if visualize:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(heat_map_list) + 2,
            figsize=(13, 4),
            gridspec_kw={"width_ratios": [1, 0.1] + [1] * len(heat_map_list)},
        )
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        axes[0].imshow(probe_img.permute(1, 2, 0))
        axes[0].axis("off")
        axes[1].axis("off")
        for i, ax in enumerate(axes[2:]):  # Adjust index to skip the spaced column
            #     for j, ax in enumerate(ax_row):
            heat_map = heat_map_list[i]
            viz.visualize_image_attr(
                heat_map.permute(1, 2, 0).numpy(),
                probe_img.permute(1, 2, 0).numpy(),
                method="blended_heat_map",
                sign="all",
                cmap="seismic",
                alpha_overlay=0.7,
                use_pyplot=False,
                plt_fig_axis=(fig, ax),
            )
            ax.set_xlabel(f"'{feat_list[i]}' head: \n" + ", \n".join(head_list[i]), fontsize=15)
            ax.set_title(feat_inst_list[i], fontsize=17)

        plt.tight_layout()

        plt.savefig(
            f"./saved_plots/{model_descr}_{probe_img_name}_{','.join(feat_list)}_tok_heatmap_viz.pdf",
            bbox_inches="tight",
            dpi=300,
        )
