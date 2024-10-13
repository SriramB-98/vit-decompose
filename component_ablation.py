from matplotlib.ticker import MultipleLocator

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

with open("./imagenet_classes.txt", "r") as fp:
    classes = [x.strip() for x in fp.readlines()]

with open("./templates.txt", "r") as fp:
    templates = [x.strip() for x in fp.readlines()]

imgnet_path = None # replace with path to ImageNet dataset

num_workers = 4 * torch.cuda.device_count()
num_batches = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

linecolors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta"]
model_keys = ["DeiT", "CLIP", "DINO", "DINOv2", "SWIN", "MaxVit"]

def get_exec_order(comp_names):
    mlp_inds = [i for i, x in enumerate(comp_names) if "mlp" in x]
    attn_inds = [i for i, x in enumerate(comp_names) if "attn" in x]
    conv_inds = [i for i, x in enumerate(comp_names) if "conv" in x]
    init_ind = comp_names.index("init")
    order = list(zip(mlp_inds, attn_inds)) 
    if conv_inds:
        new_order = []
        for i, c in enumerate(conv_inds):
            new_order.extend(order[2*i:2*i+2])
            new_order.append([c])
        order = new_order
    flattened_order = [x for y in order for x in y]
    return [init_ind] + flattened_order

plt.figure(figsize=(10, 5))

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

for lc, model_key in zip(linecolors, model_keys):
    pht = "clip_zeroshot" if model_key == "CLIP" else "imgnet_trained"
    model, model_descr, batch_size, pred_head = load_model(model_key, device, classes, templates, pred_head_type=pht)

    if model_key == "SWIN":
        detach_block, end_block = (0, 0), (3, 3)
    elif model_key == "MaxVit":
        detach_block, end_block = (0,0), (3,2)
    else:
        detach_block, end_block = 0, 12
    model.detach_from_res(detach_block, end_block)
    model.freeze_blocks(0, detach_block)
    model.to(device)
    _ = model.eval()

    dataset = ImageNet(imgnet_path, split="val", transform=model.preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    comp_names, embeds, labels = get_decomposed_embeds(
        model,
        dataloader,
        num_batches,
        device,
        heads=False,
        load_file=f"./saved_outputs/{model_descr}_imgnet_layer_decomposed_embeds.pt",
    )

    reverse_exec_order = get_exec_order(comp_names)
    print(reverse_exec_order)
    embeds = embeds[reverse_exec_order]
    comp_names = [comp_names[i] for i in reverse_exec_order]

    N = len(embeds)
    acc_list = []

    for k in range(1, N):
        with torch.no_grad():
            abl_embeds = embeds.clone()
            abl_embeds[:k] = abl_embeds[:k].mean(dim=1, keepdims=True)
            hits = pred_head(abl_embeds.sum(0).to(device)).argmax(-1).cpu() == labels[:, 0]
            acc_list.append(hits.float().mean().item())

    layer_inds = [x.strip("layer:, attnsmlpconv") for x in comp_names[1:]]
    x_ticks = [layer_inds.index(i) for i in list(set(layer_inds))]

    plt.plot(acc_list, color=lc, label=model_key)
    plt.plot(x_ticks, [acc_list[i] for i in x_ticks], "o", markerfacecolor="none", markeredgecolor=lc)

plt.grid(True, linestyle=":", color="gray")
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(2))  # Set major ticks interval for x-axis
ax.yaxis.set_major_locator(MultipleLocator(0.1))  # Set major ticks interval for y-axis
plt.xlabel("Number of layers ablated (starting from the last layer)")
plt.xlim(left=0)
plt.ylabel("ImageNet accuracy")
plt.legend()

plt.savefig("./saved_plots/ablation_plot.pdf", format="pdf", bbox_inches="tight")
