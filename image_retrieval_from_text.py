import pickle

from torchvision.datasets import ImageNet
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from helpers.linear_decompose import *
from helpers.inspect_utils import *
from helpers.utils import *
from helpers.model_utils import *
from helpers.decompose_utils import *
from helpers.interpret_utils import *

set_seed(0)

imgnet_path = None # replace with path to ImageNet dataset
if imgnet_path is None:
    raise ValueError("Please provide path to ImageNet dataset")

model_keys = ["DeiT", "CLIP", "DINO", "DINOv2", "SWIN", "MaxVit"] #

# feat = "color"
# probe_texts = ["black color", "white color"]

feat = "location"
probe_texts = ["green forest", "beach"]

# feat = "animal"
# probe_texts = ['cat', 'dog']

num_rows = 3
num_comps = 3
visualize = True


def get_imgs_across_heads(score_list, num_imgs=6):
    img_list = []
    for i, scores in enumerate(score_list):
        inds = torch.argsort(scores, descending=True)[:num_imgs].tolist()
        img_list.append(torch.stack([dataset[i][0] for i in inds]))
    return torch.stack(img_list)

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
    'animal': ['camel', 'elephant', 'giraffe', 'cat', 'dog', 'zebra', 'cheetah'],
    "person": ["face", "head", "man", "woman", "human", "arms", "legs"],
    "location": ["sea", "beach", "forest", "desert", "city", "sky", "marsh"],
    "pattern": ["spotted pattern", "striped pattern", "polka dot pattern", "plain pattern", "checkered pattern"],
    "shape": ["triangular shape", "rectangular shape", "circular shape", "octagon"],
}

feat_desc = feat_desc_dict[feat]

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


    feat_embeds = get_clip_text_embeds(clip_model, feat_desc, templates, device).weight.data.cpu()

    variations = variance_attributed(clip_aligned_embeds_decomp, feat_embeds)
    # variations = get_var_decomp(clip_aligned_embeds_decomp, feat_embeds)
    sorted_heads = torch.argsort(variations, descending=True)

    probe_vecs = get_clip_text_embeds(clip_model, probe_texts, templates, device).weight.data.cpu()

    img_list_per_probe = []
    for probe_vec in probe_vecs:
        var_scores_list = []
        var_head_index_list = []
        for start_i in range(0, len(clip_aligned_embeds_decomp), num_comps):
            var_head_index = sorted_heads[start_i : start_i + num_comps]
            var_head_index_list.append(var_head_index)
            with torch.no_grad():
                var_scores_list.append(clip_aligned_embeds_decomp[var_head_index].sum(0) @ probe_vec)
        img_list_per_probe.append(get_imgs_across_heads(var_scores_list, num_imgs=num_rows))
    img_list_per_probe = torch.cat(img_list_per_probe, dim=1)

    if visualize:
        ncols, nrows = img_list_per_probe.shape[:2]

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows)
        )  # Adjust the figsize based on the actual size of your images
        for i in range(nrows):
            for j in range(ncols):
                axs[i, j].imshow(img_list_per_probe[j][i].permute(1, 2, 0))  # Display image at position (i, j)
                axs[i, j].axis("off")  # Turn off axis numbers and ticks
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Adjust spacing between images

        plt.savefig(
            f"./saved_plots/{model_descr}_image_retrieval_from_text_{feat}-{','.join(probe_texts)}_visualization.pdf",
            format="pdf",
            bbox_inches="tight",
        )
