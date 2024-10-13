from torchvision.datasets import ImageNet
import torch
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import DataLoader

from helpers.linear_decompose import *
from helpers.inspect_utils import *
from helpers.utils import *
from helpers.model_utils import *
from helpers.decompose_utils import *
from helpers.interpret_utils import *

set_seed(0)


model_keys = ["DeiT", "CLIP", "DINO", "DINOv2", "SWIN", "MaxVit"]


# probe_img_name = 'striped_shirt_beach' #'handbag' #striped_shirt_beach
# feats = ['person', 'pattern', 'location'] 

probe_img_name = 'handbag' #striped_shirt_beach
feats = ['color', 'pattern', 'fabric'] 
imgnet_path = None # replace with path to ImageNet dataset

num_batches = 100
num_imgs = 3
visualize = True

def get_imgs_across_heads(score_list, num_imgs=6):
    img_list = []
    for i, scores in enumerate(score_list):
        inds = torch.argsort(scores, descending=True)[:num_imgs].tolist()
        img_list.append(torch.stack([dataset[i][0] for i in inds]))
    return torch.stack(img_list)


def plot_img_probe_retrieval(probe_img, closest_imgs, feats, save_path=None):
    nrows, ncols = len(closest_imgs), len(closest_imgs[0])
    fig, axes = plt.subplots(nrows=nrows+1, ncols=ncols, figsize=(7, 10),
                             gridspec_kw={'height_ratios': [1.1] + [1]*nrows})
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Plot the probe image in the first row
    axes[0, ncols//2].imshow(probe_img.permute(1, 2, 0))
    for j in range(ncols):
        axes[0, j].axis('off')

    # Plot the closest images in the subsequent rows
    for i in range(1, nrows+1):
        for j in range(ncols):
            axes[i, j].imshow(closest_imgs[i-1][j].permute(1, 2, 0))
            if j == 0:
                print(feats[i-1])
                axes[i, j].set_ylabel(feats[i-1], fontsize=17)
            axes[i, j].xaxis.set_visible(False)
            plt.setp(axes[i, j].spines.values(), visible=False)
            axes[i, j].tick_params(left=False, labelleft=False)
            axes[i, j].patch.set_visible(False)

    # Add a title on the y-axis for the second column of images
    fig.add_subplot(111, frame_on=False)  # Add a big subplot for common labels
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)  # Hide ticks and labels

    # Adjust layout
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


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
    "fabric": ["linen", "velvet", "cotton", "silk", "chiffon"]
}

num_comps_dict = {
    "MaxVit": {
        'color':12, 
        'pattern':8, 
        'fabric':9,
        'texture':9,
        'person':9,
        'location':9,
        'shape':3
    },
    "SWIN": {
        'color':3, 
        'pattern':12, 
        'fabric':9,
        'texture':9,
        'person':5,
        'location':9,
        'shape':8
    },
    "DeiT":{
        'color':3, 
        'pattern':3, 
        'fabric':3,
        'texture':3,
        'person':3,
        'location':3,
        'shape':3
    },
    "DINO":{
        'color':2, 
        'pattern':5, 
        'fabric':4,
        'texture':3,
        'person':3,
        'location':3,
        'shape':3
    },
    "DINOv2":{
        'color':1, 
        'pattern':1, 
        'fabric':4,
        'texture':3,
        'person':3,
        'location':3,
        'shape':3
    },
    "CLIP":{
        'color':3, 
        'pattern':3, 
        'fabric':4,
        'texture':3,
        'person':3,
        'location':3,
        'shape':3
    }
}


with open("./imagenet_classes.txt", "r") as fp:
    classes = [x.strip() for x in fp.readlines()]

with open("./templates.txt", "r") as fp:
    templates = [x.strip() for x in fp.readlines()]


num_workers = 4 * torch.cuda.device_count()
gpu_size = 512 * torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Beginning image retrieval from image')

for model_key in model_keys:
    print(model_key)
    pht = "clip_zeroshot" if model_key == "CLIP" else "imgnet_trained"
    model, model_descr, batch_size, pred_head = load_model(model_key, device, classes, templates, pred_head_type=pht)
    if model_key == "SWIN":
        detach_block, end_block = (2, 14), (3,2)
    elif model_key == "MaxVit":
        detach_block, end_block = (2,3), (3,2)
    else:
        detach_block, end_block = 7, 12

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
        num_batches,
        device,
        load_file=f"./saved_outputs/{model_descr}_imgnet_decomposed_embeds.pt",
    )

    with torch.no_grad():
        clip_aligned_embeds_decomp = clip_aligner_head(embeds_decomp)


    probe_img = model.preprocess(Image.open(f'./probe_imgs/{probe_img_name}.jpg'))
    decomp_model = head_outputs(model, tensor_output=True)
    model.expand_at_points(heads=True, tokens=False)
    probe_vec = decomp_model(probe_img[None,:].to(device)).cpu().squeeze(1)
    model.expand_at_points(heads=False, tokens=False)

    variations_list = []
    for feat in feats:    
        feat_embeds = get_clip_text_embeds(clip_model, feat_desc_dict[feat], templates, device).weight.data.cpu()
        variations_list.append(variance_attributed(clip_aligned_embeds_decomp, feat_embeds))
    variations = torch.stack(variations_list)

    sorted_heads_list = []

    for i in range(len(variations)):
        variations_2 = variations.clone()
        variations_2[i] = variations_2[i]*0
        sorted_heads_list.append(torch.argsort(variations[i] - variations_2.max(dim=0).values, descending=True))
    
    num_comps_list = [num_comps_dict[model_key][feat] for feat in feats]

    var_scores_list = []
    for sorted_heads, num_comps in zip(sorted_heads_list, num_comps_list):
        var_scores = F.normalize(embeds_decomp[sorted_heads[:num_comps]].sum(0))\
                    @F.normalize(probe_vec[sorted_heads[:num_comps]].sum(0), dim=-1).T
        var_scores_list.append(var_scores)

    closest_imgs = get_imgs_across_heads(var_scores_list, num_imgs=num_imgs)
    plot_img_probe_retrieval(probe_img, closest_imgs, feats, 
                         save_path=f"./saved_plots/{model_descr}_{probe_img_name}_{','.join(feats)}_img_probe_viz.pdf")
