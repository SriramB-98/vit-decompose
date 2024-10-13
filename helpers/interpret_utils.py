import torch
import numpy as np
import einops
from helpers.model_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad
def score_images(clip_model, dataloader, texts, templates, n_batches=100):
    text_embeds = get_clip_text_embeds(clip_model, texts, templates, device).weight.data
    clip_scores = []
    if n_batches is None:
        n_batches = len(dataloader)
    for i, batch in tqdm(enumerate(dataloader), total=n_batches):
        clip_embeds = clip_model(batch[0].to(device))
        clip_scores.append( (clip_embeds@text_embeds.T).cpu() )
        if i+1 == n_batches:
            break
    clip_scores = torch.cat(clip_scores, dim=0)
    return clip_scores

## TextSpan
@torch.no_grad()
def replace_with_iterative_removal(data, text_features, texts, iters, rank, device):
    results, stds = [], []
    if rank is not None:
        u, s, vh = np.linalg.svd(data, full_matrices=False)
        vh = vh[:rank]
        text_features = (
            vh.T.dot(np.linalg.inv(vh.dot(vh.T)).dot(vh)).dot(text_features.T).T
        )  # Project the text to the span of W_OV
        text_features = torch.from_numpy(text_features).float()
    
    text_features = text_features.to(device)
    data = data.to(device)
    mean_data = data.mean(dim=0, keepdim=True)
    data = data - mean_data
    reconstruct = einops.repeat(mean_data, "A B -> (C A) B", C=data.shape[0])
    reconstruct = reconstruct.detach().cpu().numpy()
    
    for i in range(iters):
        projection = data @ text_features.T
        projection_std = projection.std(axis=0).detach().cpu().numpy()
        top_n = np.argmax(projection_std)
        results.append(texts[top_n])
        stds.append(projection_std[top_n])
        text_norm = text_features[top_n] @ text_features[top_n].T
        reconstruct += (
            (
                (data @ text_features[top_n] / text_norm)[:, np.newaxis]
                * text_features[top_n][np.newaxis, :]
            )
            .detach()
            .cpu()
            .numpy()
        )
        data = data - (
            (data @ text_features[top_n] / text_norm)[:, np.newaxis]
            * text_features[top_n][np.newaxis, :]
        )
        text_features = (
            text_features
            - (text_features @ text_features[top_n] / text_norm)[:, np.newaxis]
            * text_features[top_n][np.newaxis, :]
        )
    return reconstruct, results, stds