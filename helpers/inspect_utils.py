from functools import partial
import torch
from collections import defaultdict

def my_getattr(m, x, y=None):
    xis = x.split('.')
    t = m
    for xi in xis:
        if xi.isnumeric():
            t = t[int(xi)]
        else:
            t = getattr(t, xi, y)
    return t

def read(activation_dict, inp, out, name, read_mode):
    if read_mode == 'inp':
        x = inp
    elif read_mode == 'out':
        x = out
#     print(x)
    activation_dict[name] = x#.clone()
    return

def get_hook(name, activation_dict, read_mode=None, modify_fn=None):
    # the hook signature
    def hook(model, input, output):
        if read_mode is not None:
            read(activation_dict, input, output, name, read_mode)
        if modify_fn is not None:
            output = modify_fn(activation_dict, input, output, name)
#             read_fn(input, output, name+"_after")
            return output
    return hook