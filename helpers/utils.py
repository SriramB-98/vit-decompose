import time
import torch
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import hsv_to_rgb

def update_time(t):
    print(f"time elapsed: {time.time() - t}")
    return time.time()

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dict_to_str(dic, p=4):
    s = ''
    for k, it in dic.items():
        if not isinstance(it, str):
            it = np.format_float_positional(it, precision=p, fractional=False)
        s += str(k)+'='+it+'_'
    return s[:-1]

def str_to_dict(s, p=4):
    dic = {}
    for it in s.split('_'):
        k, v = it.split('=')
        dic[k] = float(np.format_float_positional(float(v), precision=p, fractional=False))
    return dic

def list_to_str(lis, sep='\n'):
    lis = [str(x) for x in lis]
    return sep.join(lis)

def expand_and_flatten(model_list):
    if not model_list:
        return []
    flattened_list = []
    for model in model_list:
        child_list = expand_and_flatten(list(model.children()))
        if not child_list and list(model.parameters()):
            child_list = [model]
        flattened_list.extend(child_list)
    return flattened_list

def to_numpy(tens):
    return tens.cpu().detach().numpy()

def colorgrid(array, norm=0.1):
    red = np.zeros(shape=array.shape)
    blue = 240/360*np.ones(shape=array.shape)
    hue = red*(array > 0) + blue*(array<=0)
    sat = np.clip(np.abs(array)/norm, -1, 1)
    val = np.ones(shape=array.shape)
    hsv_img = np.stack((hue, sat, val), axis=-1)
#     print(hsv_img.shape, hsv_img)
#     rgb_img = cv2.cvtColor(hsv_img.astype(np.float32), cv2.COLOR_HSV2BGR)
    return hsv_to_rgb(hsv_img), hsv_img

# def reload_module(module_name):
#     import importlib
#     import module_name
#     importlib.reload(module_name)
#     from module_name import *