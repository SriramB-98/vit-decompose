import torch
from torch.nn import functional as F
from torch.linalg import norm
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm.auto import tqdm

def avg_proj_comp(vecs, k, num_iters=100):
    proj_comps = []
    for _ in range(num_iters):
        inds = np.random.choice(len(vecs), size=k, replace=False)
        bases_vecs, probe_vec = vecs[inds[:-1]], vecs[inds[-1]]
        bases_vecs = torch.qr(bases_vecs.T).Q.T
        proj_comps.append(norm(bases_vecs.T @ bases_vecs @ F.normalize(probe_vec, dim=0)).item())
    return np.mean(proj_comps)

def project_out(decomp, head_list, vec_list):
    new_decomp = decomp.clone()
    for head, vec in zip(head_list, vec_list):
        vec = torch.qr(vec.T).Q.T
        if head is None:
            new_decomp = new_decomp - (new_decomp@vec.T)@(vec)
        else:
            new_decomp[:,head] = new_decomp[:,head] - (new_decomp[:,head]@vec.T)@(vec)
    return new_decomp

def preds_on_imgnet(label_filter, head_list, vec_list, pred_head, path=None):
    preds_list = []
    labels_list = []
    for i in tqdm(range(8)):
        try:
            labels, _, decomp = torch.load(f'{path}/head_decomp_{i}.pkl')
        except FileNotFoundError as e:
            print(e)
            break
        mask = [(l.item() in label_filter) for l in labels]
        labels, decomp = labels[mask], decomp[mask]
        if len(labels) == 0: continue
        proj_decomp = project_out(decomp, head_list, vec_list)
        preds_list.append(pred_head(proj_decomp.sum(1).to(device)).cpu())
        labels_list.append(labels)
    preds = torch.cat(preds_list)
    labels = torch.cat(labels_list)
    print('Accuracy:', (preds.argmax(-1) == labels).float().mean().item())
    return preds, labels

def classwise_imgnet_acc(preds, labels, metric='conf'):
    if metric == 'acc':
        preds = preds.argmax(-1)
    elif metric == 'conf':
        preds = torch.softmax(preds, -1)
    label_set = set(labels.tolist())
    acc_dict = dict()
    for label in label_set:
        if metric == 'acc':
            scores = (preds[labels == label] == label)
        elif metric == 'conf':
            scores = (preds[labels == label,label])
        acc_dict[label] = scores.float().mean().item()
    return acc_dict