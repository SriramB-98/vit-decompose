import gc
from itertools import chain
from helpers.model_utils import *
from helpers.decompose_utils import *
from torchvision.datasets import ImageNet

import torch
from torchvision import transforms as T
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm import create_model

import clip
import open_clip
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils.parametrizations import orthogonal
from tqdm.auto import tqdm

error_dump = []
imgnet_path = None # replace with path to ImageNet dataset

def list_to_tensor(l):
    if type(l) is torch.Tensor:
        return l
    elif type(l) is list:
        return torch.stack([list_to_tensor(i) for i in l])


def detach_input(model, inp):
    det_args = inp[0].detach()
    det_args.requires_grad_(True)
    return det_args


def collect_layer_components(decomp):
    layer_comps = []
    while type(decomp) is list:
        layer_comps.append(decomp[1])
        decomp = decomp[0]
    return decomp, layer_comps


def apply_head(head, decomp, num_comps=None):
    head_weight, head_bias = head.weight.data.cpu(), head.bias.data.cpu()
    if head_bias is None:
        head_bias = 0
    if num_comps is None:
        num_comps = decomp.shape[1]
    out = decomp @ head_weight.T + head_bias / num_comps
    return out


class Aligner(torch.nn.Module):
    def __init__(self, num_heads, model_dim, clip_dim, reduce=True):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(num_heads, model_dim, clip_dim) * 0.02)
        self.bias = torch.nn.Parameter(torch.randn(clip_dim) * 0.02)
        self.reduce = reduce

    def forward(self, x):
        # x.shape = (num_heads, batch, model_dim)
        out = x @ self.weights + self.bias / len(x)
        if self.reduce:
            out = out.sum(0)
        return out


class OrthonormalAligner(torch.nn.Module):
    def __init__(self, num_heads, model_dim, clip_dim, reduce=True):
        super().__init__()
        self.ortholinear = torch.nn.ModuleList(
            [orthogonal(torch.nn.Linear(model_dim, clip_dim)) for _ in range(num_heads)]
        )
        self.bias = torch.nn.Parameter(torch.randn(clip_dim) * 0.02)
        self.reduce = reduce

    def forward(self, x):
        # x.shape = (num_heads, batch, model_dim)
        weights = torch.stack([m.weight for m in self.ortholinear], dim=0)
        out = x @ weights + self.bias / len(x)
        if self.reduce:
            out = out.sum(0)
        return out


class MultipleMLPAligner(torch.nn.Module):
    def __init__(self, num_heads, model_dim, clip_dim, hidden_dim=1000, reduce=True):
        super().__init__()
        self.mlps = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(model_dim, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, clip_dim)
                )
                for _ in range(num_heads)
            ]
        )
        self.reduce = reduce

    def forward(self, x):
        # x.shape = (num_heads, batch, model_dim)
        out = [mlp(xi) for mlp, xi in zip(self.mlps, x)]
        out = torch.stack(out, dim=0)
        if self.reduce:
            out = out.sum(0)
        return out


class MLPAligner(torch.nn.Module):
    def __init__(self, model_dim, clip_dim, hidden_dim=1000, reduce=True):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, clip_dim)
        )
        self.reduce = reduce

    def forward(self, x):
        # x.shape = (num_heads, batch, model_dim)
        out = self.mlp(x)
        if self.reduce:
            out = out.sum(0)
        return out


def align_with_clip(
    model,
    clip_model,
    clip_name,
    clip_type,
    dataloader,
    device,
    composite=False,
    orthonormal=False,
    mlp=False,
    load_file=None,
    retrain=False,
    **train_kwargs,
):
    try:
        if load_file is not None:
            if orthonormal:
                (num_heads, model_dim, clip_dim), aligner_state_dict = torch.load(load_file, map_location=device)
                aligner = OrthonormalAligner(num_heads, model_dim, clip_dim)
                aligner.load_state_dict(aligner_state_dict)
                aligner.to(device)
            else:
                aligner = torch.load(load_file, map_location=device)
            if not retrain:
                return aligner
            else:
                print(f"Retraining from {load_file}")
        else:
            aligner = None
    except FileNotFoundError:
        print("File not found, training from scratch!")
        aligner = None

    lr = train_kwargs.get("lr", 1e-3)
    wd = train_kwargs.get("wd", 1e-6)
    epochs = train_kwargs.get("epochs", 3)
    break_at_loss = train_kwargs.get("break_at_loss", 0.1)
    wait_for_batches = train_kwargs.get("wait_for_batches", 10)
    loss_type = train_kwargs.get("loss", "cos_sim_loss")
    orthonormalize = train_kwargs.get("orthonormalize", 0)
    alpha = train_kwargs.get("alpha", 1)
    if clip_type == "openai":
        tokenizer = clip.tokenize
    elif clip_type == "openclip":
        tokenizer = open_clip.get_tokenizer(clip_name.replace("/", "-"))
    else:
        raise ValueError('clip_type must be one of "openai" or "openclip"')

    for batch in dataloader:
        model_shape = model(batch[0][0][:1].to(device)).shape
        clip_shape = clip_model(batch[0][1][:1].to(device)).shape
        break
    clip_dim = clip_shape[1]
    clip_model, normalizer = clip_model.clip_model, clip_model.normalize
    clip_model.to(device)
    clip_model.eval()

    if aligner is None:
        if composite:
            num_heads, model_dim = model_shape[0], model_shape[2]
            print("Num heads, model_dim", num_heads, model_dim)
            if orthonormal:
                aligner = OrthonormalAligner(num_heads, model_dim, clip_dim)
            elif mlp:
                aligner = MLPAligner(model_dim, clip_dim)
            else:
                aligner = Aligner(num_heads, model_dim, clip_dim)
            aligner.to(device)
        else:
            model_dim = model_shape[1]
            aligner = torch.nn.Linear(model_dim, clip_dim, bias=True)
            aligner.to(device)

    optimizer = torch.optim.Adam(
        aligner.parameters(),
        lr=lr,
        weight_decay=wd,
    )

    min_loss = 10000.0
    batches_without_progress = 0
    labels = None
    other_metrics = []
    loss_metrics = []
    for e in range(epochs):
        for i, batch in enumerate(dataloader):
            imgs = batch[0][0].to(device)
            clip_imgs = batch[0][1].to(device)

            with torch.no_grad():
                model_img_features = model(imgs)
            aligned_features = aligner(model_img_features)
            aligned_features = F.normalize(aligned_features, dim=-1)

            with torch.no_grad():
                clip_img_features = clip_model.encode_image(normalizer(clip_imgs))
                clip_img_features = F.normalize(clip_img_features, dim=-1).to(torch.float32)

            if loss_type == "contrastive_loss":
                with torch.no_grad():
                    texts = tokenizer(batch[1]).to(device)
                    text_embeddings = clip_model.encode_text(texts)
                    text_embeddings = F.normalize(text_embeddings, dim=-1)
                #                     clip_img_features = clip_model.encode_image(normalizer(imgs))
                #                     clip_img_features = F.normalize(clip_img_features, dim=-1).to(torch.float32)
                #                     aligned_features = clip_img_features

                logits = aligned_features @ text_embeddings.T
                if labels is None:
                    labels = torch.arange(logits.shape[0]).to(device)
                loss = alpha * 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
                img_loss = (1 - F.cosine_similarity(aligned_features, clip_img_features)).mean()
                loss += img_loss
                acc = 0.5 * ((logits.argmax(-1) == labels).float().mean() + (logits.argmax(0) == labels).float().mean())
                other_metrics = [acc.item(), img_loss.item()]
            elif loss_type == "cos_sim_loss":
                loss = (1 - F.cosine_similarity(aligned_features, clip_img_features)).mean()
                loss_metrics.append(loss.item())
                if orthonormalize != 0:
                    weight = getattr(aligner, "weights", getattr(aligner, "weight", None))
                    model_dim = weight.shape[1]
                    clip_dim = weight.shape[2]
                    orth_loss = clip_dim * torch.mean(
                        (weight.transpose(-1, -2) @ weight - torch.eye(clip_dim, clip_dim).to(device)) ** 2
                    )
                    loss += orthonormalize * orth_loss
                    other_metrics = [orth_loss.item()]
            elif loss_type == "mse_loss":
                loss = F.mse_loss(aligned_features, clip_img_features)
            else:
                raise ValueError

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            new_min_loss = min(min_loss, loss.item())
            if i % 10 == 0:
                print(f"Epoch {e}, batch {i}:", loss.item(), *other_metrics)
                if loss_metrics:
                    print('Avg loss', np.mean(loss_metrics) )

            if new_min_loss < min_loss:
                min_loss = new_min_loss
                batches_without_progress = 0
            else:
                batches_without_progress += 1
            if loss.item() <= break_at_loss or batches_without_progress == wait_for_batches:
                break
        if loss.item() <= break_at_loss or batches_without_progress == wait_for_batches:
            break
        print(f"Epoch {e} loss:", loss.item())

    if load_file is not None:
        if orthonormal:
            torch.save(((num_heads, model_dim, clip_dim), aligner.state_dict()), load_file)
        else:
            torch.save(aligner, load_file)

    return aligner


def get_clip_text_embeds(clip_model, classes, templates, device, load_file=None):
    clip_model, clip_name, clip_type = clip_model.clip_model, clip_model.clip_model_name, clip_model.clip_model_type
    if clip_type == "openai":
        tokenizer = clip.tokenize
    elif clip_type == "openclip":
        tokenizer = open_clip.get_tokenizer(clip_name.replace("/", "-"))
    else:
        raise ValueError('clip_type must be one of "openai" or "openclip"')

    clip_model.to(device).eval()
    try:
        if load_file is not None:
            embeds = torch.load(load_file, map_location=device)
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        with torch.no_grad():
            embeds = []
            for classname in tqdm(classes):
                texts = [template.format(classname) for template in templates]  # format with class
                texts = tokenizer(texts).to(device)
                class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                embeds.append(class_embedding)
        embeds = torch.stack(embeds, dim=1).to(device)
        if load_file is not None:
            torch.save(embeds, load_file)

    head = torch.nn.Linear(embeds.shape[0], embeds.shape[1], bias=False)
    head.weight.data = embeds.to(torch.float32).T
    return head


def get_head(model, dataloader, num_classes, device, bias=True, load_file=None, retrain=False, **train_kwargs):
    try:
        if load_file is not None:
            head = torch.load(load_file, map_location=device)
        if not retrain:
            return head
    except FileNotFoundError:
        print("File not found, training from scratch!")
        head = None

    for batch in dataloader:
        model_shape = model(batch[0][:1].to(device)).shape
        break

    if head is None:
        head = torch.nn.Linear(model_shape[1], num_classes, bias=bias)
    head.to(device)

    lr = train_kwargs.get("lr", 1e-3)
    wd = train_kwargs.get("wd", 1e-6)
    epochs = train_kwargs.get("epochs", 3)
    stop_batches = train_kwargs.get("stop_batches", 1e8)
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=lr,
        weight_decay=wd,
    )
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    for e in range(epochs):
        for i, batch in enumerate(dataloader):
            imgs, labels = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                features = model(imgs)
            logits = head(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {e}, batch {i}:", loss.item())
            if i >= stop_batches:
                break
        print(f"Epoch {e} loss:", loss.item())
    if load_file is not None:
        torch.save(head, load_file)
    return head


def test_model(model, head, dataloader, device, masking_fn=None):
    hits, total = 0, 0
    for batch in tqdm(dataloader):
        inp = batch[0].to(device)
        if masking_fn:
            inp = masking_fn(inp, batch[3].to(device))
        with torch.no_grad():
            preds = head(model(inp)).cpu()
        hits += (preds.argmax(-1) == batch[1]).float().sum().item()
        total += len(batch[1])
    return {"acc": hits / total}


class TimmMaxVitModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = create_model(model_name, pretrained=True)
        transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        self.normalizer = transform.transforms[-1]
        self.preprocess = T.Compose(transform.transforms[:-1])
        self.expand_dict = None
        self.start_block = 0
        self.start_stage = 0

    def forward(self, x):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            x = self.normalizer(x)
            out = self.model.forward_features(x)
        return out.mean(dim=(-1, -2))

    def detach_from_res(self, start_stage_block, end_stage_block):
        # assuming that end_stage_block is last block
        start_stage, start_block = start_stage_block
        self.start_stage, self.start_block = start_stage, start_block
        for si, stage_l in enumerate(self.model.stages[start_stage:]):
            sb = start_block if si == 0 else 0
            for bi, block_l in enumerate(stage_l.blocks[sb:]):
                print(f"Detaching stage:{start_stage+si}, block:{sb+bi}")
                block_l.conv.pre_norm.register_forward_pre_hook(detach_input)
                block_l.attn_block.norm1.register_forward_pre_hook(detach_input)
                block_l.attn_block.norm2.register_forward_pre_hook(detach_input)
                block_l.attn_grid.norm1.register_forward_pre_hook(detach_input)
                block_l.attn_grid.norm2.register_forward_pre_hook(detach_input)

        self.model.stages[start_stage].blocks[start_block].register_forward_pre_hook(detach_input)
        return

    def expand_at_points(self, heads=True, tokens=True):
        if self.expand_dict is None:

            self.expand_dict = {"heads": heads, "tokens": tokens}

            def expand_heads(model, inp, out):
                if self.expand_dict["heads"]:
                    try:
                        out.grad_fn.metadata["expand"] = {1, 2, 3}
                    except AttributeError:
                        pass
                return out

            def expand_toks(model, inp, out):
                if self.expand_dict["tokens"]:
                    try:
                        out.grad_fn.next_functions[0][0].metadata["expand"] = True
                    except AttributeError:
                        pass
                return out

            for si, stage_l in enumerate(self.model.stages[self.start_stage :]):
                sb = self.start_block if si == 0 else 0
                for bi, block_l in enumerate(stage_l.blocks[sb:]):
                    # print('Expanding stage:', self.start_stage+si, 'block:', sb+bi)
                    block_l.attn_block.attn.probe_h.register_forward_hook(expand_heads)
                    block_l.attn_block.attn.probe_t.register_forward_hook(expand_toks)
                    block_l.attn_grid.attn.probe_h.register_forward_hook(expand_heads)
                    block_l.attn_grid.attn.probe_t.register_forward_hook(expand_toks)

        else:
            self.expand_dict["heads"] = heads
            self.expand_dict["tokens"] = tokens

    def freeze_blocks(self, start_stage_block, end_stage_block):
        # assuming that start_stage_block = (0, 0)
        stage, block = end_stage_block

        for p in self.model.stem.parameters():
            p.requires_grad_(False)

        for si, stage_l in enumerate(self.model.stages):
            if si < stage:
                for p in stage_l.parameters():
                    p.requires_grad_(False)
            elif si == stage:
                # for p in stage_l.downsample.parameters():
                #     p.requires_grad_(False)
                for bi, block_l in enumerate(stage_l.blocks):
                    if bi == block:
                        return
                    else:
                        for p in block_l.parameters():
                            p.requires_grad_(False)
        return

    def collect_components(self, decomp):
        self.init, self.layer_comps = collect_layer_components(decomp)
        self.conv_comps, self.attn_comps, self.mlp_comps = [], [], []
        for i, comp in enumerate(self.layer_comps):
            if (i + 1) % 5 == 0:
                self.conv_comps.append(comp)
            elif (i + 1) % 5 in [1, 3]:
                self.mlp_comps.append(comp)
            else:
                self.attn_comps.append(comp)

    def delete_components(self):
        del self.init
        del self.layer_comps, self.attn_comps, self.mlp_comps
        gc.collect()
        return

    def get_mlp_component(self, layer_i):
        return self.mlp_comps[layer_i]

    def get_conv_component(self, layer_i):
        return self.conv_comps[layer_i]

    def get_attn_component(self, layer_i, head_i=None, token_i=None):
        comp = self.attn_comps[layer_i]
        comp = list_to_tensor(comp)
        expand_dims = []
        if self.expand_dict["tokens"]:
            expand_dims.append(token_i if token_i is not None else slice(None))
        if self.expand_dict["heads"]:
            expand_dims.append(head_i if head_i is not None else slice(None))
        return comp[tuple(expand_dims)]


class TimmSwinModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = create_model(model_name, pretrained=True)
        transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        self.normalizer = transform.transforms[-1]
        self.preprocess = T.Compose(transform.transforms[:-1])
        self.expand_dict = None
        self.start_block = 0
        self.start_stage = 0

    def forward(self, x):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            x = self.normalizer(x)
            out = self.model.forward_features(x)
        return self.model.forward_head(out, pre_logits=True)

    def detach_from_res(self, start_stage_block, end_stage_block):
        # assuming that end_stage_block = (0, 0)
        start_stage, start_block = start_stage_block
        self.start_stage, self.start_block = start_stage, start_block
        for si, stage_l in enumerate(self.model.layers[start_stage:]):
            sb = start_block if si == 0 else 0
            for bi, block_l in enumerate(stage_l.blocks[sb:]):
                print(f"Detaching stage:{start_stage+si}, block:{sb+bi}")
                block_l.norm1.register_forward_pre_hook(detach_input)
                block_l.norm2.register_forward_pre_hook(detach_input)
        self.model.layers[start_stage].blocks[start_block].register_forward_pre_hook(detach_input)
        return

    def expand_at_points(self, heads=True, tokens=True):
        if self.expand_dict is None:

            self.expand_dict = {"heads": heads, "tokens": tokens}

            def expand_heads(model, inp, out):
                if self.expand_dict["heads"]:
                    try:
                        out.grad_fn.metadata["expand"] = {2}
                    except AttributeError:
                        pass
                return out

            def expand_window_toks(model, inp, out):
                if self.expand_dict["tokens"]:
                    try:
                        out.grad_fn.next_functions[0][0].metadata["expand"] = True
                    except AttributeError:
                        pass
                return out

            def expand_window(model, inp, out):
                if self.expand_dict["tokens"]:
                    try:
                        out.grad_fn.metadata["expand"] = {1, 2}
                    except AttributeError:
                        pass
                return out

            for si, stage_l in enumerate(self.model.layers[self.start_stage :]):
                sb = self.start_block if si == 0 else 0
                for bi, block_l in enumerate(stage_l.blocks[sb:]):
                    block_l.attn.probe_w.register_forward_hook(expand_window_toks)
                    block_l.attn.probe_h.register_forward_hook(expand_heads)
                    block_l.probe_nw.register_forward_hook(expand_window)
        else:
            self.expand_dict["heads"] = heads
            self.expand_dict["tokens"] = tokens

    def freeze_blocks(self, start_stage_block, end_stage_block):
        # assuming that start_stage_block = (0, 0)
        stage, block = end_stage_block

        for p in self.model.patch_embed.parameters():
            p.requires_grad_(False)

        for si, stage_l in enumerate(self.model.layers):
            if si < stage:
                for p in stage_l.parameters():
                    p.requires_grad_(False)
            elif si == stage:
                for p in stage_l.downsample.parameters():
                    p.requires_grad_(False)
                for bi, block_l in enumerate(stage_l.blocks):
                    if bi == block:
                        return
                    else:
                        for p in block_l.parameters():
                            p.requires_grad_(False)
        return

    def collect_components(self, decomp):
        self.init, self.layer_comps = collect_layer_components(decomp)
        self.attn_comps = [self.layer_comps[2 * i + 1] for i in range(len(self.layer_comps) // 2)]
        self.mlp_comps = [self.layer_comps[2 * i] for i in range(len(self.layer_comps) // 2)]

    def delete_components(self):
        del self.init
        del self.layer_comps, self.attn_comps, self.mlp_comps
        gc.collect()
        return

    def get_mlp_component(self, layer_i):
        return self.mlp_comps[layer_i]

    def get_attn_component(self, layer_i, head_i=None, win_token_i=None, window_i=None):
        comp = self.attn_comps[layer_i]
        comp = list_to_tensor(comp)
        expand_dims = []
        if self.expand_dict["tokens"]:
            expand_dims.append(win_token_i if win_token_i is not None else slice(None))
        if self.expand_dict["heads"]:
            expand_dims.append(head_i if head_i is not None else slice(None))
        if self.expand_dict["tokens"]:
            expand_dims.append(window_i if window_i is not None else slice(None))
        return comp[tuple(expand_dims)]


class TimmVitModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = create_model(model_name, pretrained=True)
        transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        self.normalizer = transform.transforms[-1]
        self.preprocess = T.Compose(transform.transforms[:-1])
        self.expand_dict = None
        self.start_block = 0

    def forward(self, x):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            x = self.normalizer(x)
            out = self.model.forward_features(x)
        return self.model.forward_head(out, pre_logits=True)

    def detach_from_res(self, start_block, end_block):
        self.start_block = start_block
        self.end_block = end_block
        for p in chain(self.model.patch_embed.parameters(), self.model.norm_pre.parameters()):
            p.requires_grad_(False)

        for i in range(start_block, end_block):
            self.model.blocks[i].norm1.register_forward_pre_hook(detach_input)
            self.model.blocks[i].norm2.register_forward_pre_hook(detach_input)

        self.model.blocks[start_block].register_forward_pre_hook(detach_input)
        return

    def expand_at_points(self, heads=True, tokens=True, block_list=None):
        if self.expand_dict is None:

            self.expand_dict = {"heads": heads, "tokens": tokens}

            def expand_heads(model, inp, out):
                if self.expand_dict["heads"]:
                    try:
                        out.grad_fn.metadata["expand"] = {2}
                    except AttributeError:
                        pass
                return out

            def expand_toks(model, inp, out):
                if self.expand_dict["tokens"]:
                    try:
                        out.grad_fn.next_functions[0][0].metadata["expand"] = True
                    except AttributeError:
                        pass
                return out

            if block_list is None: block_list = range(self.start_block, self.end_block)
            for i in block_list:
                self.model.blocks[i].attn.probe_h.register_forward_hook(expand_heads)
                self.model.blocks[i].attn.probe_t.register_forward_hook(expand_toks)
        else:
            self.expand_dict["heads"] = heads
            self.expand_dict["tokens"] = tokens

    def freeze_blocks(self, start_block, end_block):
        self.model.cls_token.requires_grad_(False)
        self.model.pos_embed.requires_grad_(False)
        for p in chain(self.model.patch_embed.parameters(), self.model.norm_pre.parameters()):
            p.requires_grad_(False)

        for bi in range(start_block, end_block):
            block = self.model.blocks[bi]
            for p in block.parameters():
                p.requires_grad_(False)
        return

    def collect_components(self, decomp):
        self.init, self.layer_comps = collect_layer_components(decomp)
        self.attn_comps = [self.layer_comps[2 * i + 1] for i in range(len(self.layer_comps) // 2)]
        self.mlp_comps = [self.layer_comps[2 * i] for i in range(len(self.layer_comps) // 2)]
        self.num_layers = len(self.attn_comps)
        self.num_heads = [len(comp) for comp in self.attn_comps]

    def delete_components(self):
        del self.init
        del self.layer_comps, self.attn_comps, self.mlp_comps
        gc.collect()
        return

    def get_mlp_component(self, layer_i):
        return self.mlp_comps[layer_i]

    def get_attn_component(self, layer_i, head_i=None, token_i=None):
        comp = self.attn_comps[layer_i]
        comp = list_to_tensor(comp)
        expand_dims = []
        if self.expand_dict["tokens"]:
            expand_dims.append(token_i if token_i is not None else slice(None))
        if self.expand_dict["heads"]:
            expand_dims.append(head_i if head_i is not None else slice(None))
        return comp[tuple(expand_dims)]


class CLIPModel(torch.nn.Module):
    def __init__(self, model_type, model_name, pretrained=None):
        self.clip_model_name = model_name
        self.clip_model_type = model_type
        super().__init__()
        if model_type == "openai":
            self.clip_model, preprocess = clip.load(model_name, device="cpu")
        elif model_type == "openclip":
            self.clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.normalize = preprocess.transforms[-1]
        self.preprocess = T.Compose(preprocess.transforms[:-1])
        self.expand_dict = None
        self.start_block = 0

    def forward(self, x):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            img_features = self.clip_model.encode_image(self.normalize(x))
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features.to(torch.float32)

    def detach_from_res(self, start_block, end_block):
        self.start_block = start_block
        self.end_block = end_block
        for i in range(start_block, end_block):
            self.clip_model.visual.transformer.resblocks[i].ln_1.register_forward_pre_hook(detach_input)
            self.clip_model.visual.transformer.resblocks[i].ln_2.register_forward_pre_hook(detach_input)
        self.clip_model.visual.transformer.resblocks[start_block].register_forward_pre_hook(detach_input)
        return

    def expand_at_points(self, heads=True, tokens=True, block_list=None):
        if self.expand_dict is None:

            self.expand_dict = {"heads": heads, "tokens": tokens}

            def expand_at_attn(model, inp, out):
                if self.expand_dict["heads"]:
                    try:
                        out[0].grad_fn.next_functions[0][0].next_functions[1][0].metadata["expand"] = {1}
                    except AttributeError:
                        pass
                if self.expand_dict["tokens"]:
                    try:
                        reshape = out[0].grad_fn.next_functions[0][0].next_functions[1][0]
                        reshape.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][
                            0
                        ].metadata["expand"] = True
                    except AttributeError:
                        pass
                return out

            if block_list is None: block_list = range(self.start_block, self.end_block)
            for i in block_list:
                self.clip_model.visual.transformer.resblocks[i].attn.register_forward_hook(expand_at_attn)
        else:
            self.expand_dict["heads"] = heads
            self.expand_dict["tokens"] = tokens

    def freeze_blocks(self, start_block, end_block):
        self.clip_model.visual.class_embedding.requires_grad_(False)
        self.clip_model.visual.positional_embedding.requires_grad_(False)
        for p in chain(self.clip_model.visual.conv1.parameters(), self.clip_model.visual.ln_pre.parameters()):
            p.requires_grad_(False)

        for bi in range(start_block, end_block):
            block = self.clip_model.visual.transformer.resblocks[bi]
            for p in block.parameters():
                p.requires_grad_(False)
        return

    def collect_components(self, decomp):
        self.init, self.layer_comps = collect_layer_components(decomp)
        self.attn_comps = [self.layer_comps[2 * i + 1] for i in range(len(self.layer_comps) // 2)]
        self.mlp_comps = [self.layer_comps[2 * i] for i in range(len(self.layer_comps) // 2)]

    def delete_components(self):
        del self.init
        del self.layer_comps, self.attn_comps, self.mlp_comps
        gc.collect()
        return

    def get_mlp_component(self, layer_i):
        return self.mlp_comps[layer_i]

    def get_attn_component(self, layer_i, head_i=None, token_i=None):
        comp = self.attn_comps[layer_i]
        comp = list_to_tensor(comp)
        expand_dims = []
        if self.expand_dict["tokens"]:
            expand_dims.append(token_i if token_i is not None else slice(None))
        if self.expand_dict["heads"]:
            expand_dims.append(head_i if head_i is not None else slice(None))
        return comp[tuple(expand_dims)]


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def load_model(model_key, device, classes=None, templates=None, pred_head_type="imgnet_trained"):
    pred_head = None
    if model_key == "DINO":
        model_name = "vit_base_patch16_224.dino"
        model_descr = model_name.replace("/", "-")
        model = TimmVitModel(model_name)
        batch_size = 64
    if model_key == "DINOv2":
        model_name = "vit_base_patch14_dinov2.lvd142m"
        model_descr = model_name.replace("/", "-")
        model = TimmVitModel(model_name)
        batch_size = 8
    if model_key == "CLIP":
        model_type = "openclip"
        model_name = "ViT-B-16"
        model_descr = f"{model_type}-{model_name.replace('/','-')}"
        model = CLIPModel(model_type, model_name, pretrained="laion2b_s34b_b88k")
        batch_size = 64
    if model_key == "SWIN":
        model_name = "swin_base_patch4_window7_224"
        model_descr = model_name.replace("/", "-")
        model = TimmSwinModel(model_name)
        pred_head = model.model.head.fc
        batch_size = 32
    if model_key == "DeiT":
        model_name = "deit_base_patch16_224"
        model_descr = model_name.replace("/", "-")
        model = TimmVitModel(model_name)
        pred_head = model.model.head
        batch_size = 64
    if model_key == "MaxVit":
        model_name = "maxvit_small_tf_224.in1k"
        model_descr = model_name.replace("/", "-")
        model = TimmMaxVitModel(model_name)
        batch_size = 32
        reshape_fn = Lambda(lambda t: t.unsqueeze(dim=-1).unsqueeze(dim=-1))
        pred_head = torch.nn.Sequential(*([reshape_fn] + [p for p in model.model.head.children()][1:]))
    model = model.to(device)

    if pred_head_type == "imgnet_trained":
        if pred_head is None:
            imgnet_train_dataset = ImageNet(
                imgnet_path, split="train", transform=model.preprocess
            )
            imgnet_train_dataloader = DataLoader(
                imgnet_train_dataset, batch_size=max(16, batch_size), shuffle=True, num_workers=4
            )
            pred_head = get_head(
                model.to(device),
                imgnet_train_dataloader,
                len(classes),
                device,
                bias=True,
                retrain=False,
                load_file=f"./saved_outputs/{model_descr}_imgnet_trained_head.pt",
                epochs=3,
                stop_batches=1000 * (64 // batch_size),
                lr=3e-4,
            )
    if pred_head_type == "clip_zeroshot":
        pred_head = get_clip_text_embeds(
            model, classes, templates, device, load_file=f"./saved_outputs/{model_descr}_imgnet_classes_head.pt"
        )

    return model, model_descr, batch_size, pred_head


def get_clip_and_aligner(
    model,
    model_descr,
    device,
    clip_type="openclip",
    clip_model_name="ViT-L/14",
    pretrained="laion2b_s32b_b82k",
    bs=32,
    num_workers=4,
):
    if "clip" in model_descr:
        clip_model = model
        clip_model_descr = model_descr
        clip_aligner_head = torch.nn.Identity()
    else:
        clip_model_descr = f"{clip_type}-{clip_model_name}".replace("/", "-")
        clip_model = CLIPModel(clip_type, clip_model_name, pretrained).to(device).eval()

        imgnet_train_dataset = ImageNet(
            imgnet_path,
            split="train",
            transform=lambda t: [model.preprocess(t), clip_model.preprocess(t)],
        )
        imgnet_train_dataloader = DataLoader(imgnet_train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

        model.expand_at_points(heads=True, tokens=False)
        clip_aligner_head = align_with_clip(
            head_outputs(model, tensor_output=True),
            clip_model,
            clip_model_name,
            clip_type,
            imgnet_train_dataloader,
            device,
            retrain=False,
            load_file=f"./saved_outputs/{model_descr}_{clip_model_descr}_"
            f"composite_orthonormalized_cossim_imgnet_aligner.pt",
            orthonormalize=1.,
            composite=True,
            epochs=2,
            lr=3e-4,
            break_at_loss=0.0,
            alpha=0.2,
            wait_for_batches=50,
            loss="cos_sim_loss",
        )
        model.expand_at_points(heads=False, tokens=False)
        clip_aligner_head.reduce = False
        clip_aligner_head.cpu()
    return clip_model, clip_model_descr, clip_aligner_head
