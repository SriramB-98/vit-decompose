error_dump = []
import torch
import numpy as np
from torch.nn import functional as F
from functools import partial

def split(t, dim, chunk_size):
    assert  t.shape[dim] % chunk_size == 0
    num_chunks = t.shape[dim] // chunk_size
    split_list = [torch.narrow(t, dim, i*chunk_size,chunk_size) for i in range(num_chunks)]
    chunk_shape = split_list[0].shape
    padded_split_list = []
    zero_tensor = torch.zeros(*[1 for _ in chunk_shape])
    for i in range(num_chunks):
        init_shape, fin_shape = list(chunk_shape), list(chunk_shape)
        init_shape[dim] *= i
        fin_shape[dim] *= num_chunks - i -1
        init_pad = zero_tensor.expand(*init_shape)
        fin_pad = zero_tensor.expand(*fin_shape)
        padded_split_list.append(torch.cat([init_pad, 
                                            split_list[i], 
                                            fin_pad], dim=dim))
    return padded_split_list

def prod(axes):
    if len(axes) == 0:
        return 1
    else:
        return axes[0]*prod(axes[1:])

class Metadata:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
    
def extract_shape_metadata(tensor):
    shape = tensor.shape
    metadata = Metadata(tensor.device, tensor.dtype)
    return shape, metadata
    
def convert_to_int(i, n):
    N = 18446744073709551616
    thresh = 18446744073709551616//2
    if i < thresh:
        return i
    else:
        return i - N + n
    
def map_l(fn, l, fn2=None):
    if type(l) is torch.Tensor:
        return fn(l)
    elif type(l) is list:
        return [map_l(fn, sl, fn2) for sl in l]
    else:
        raise Exception(f'type {type(l)} is not handled')

def reduce_l(fn, l, acc=0):
    if type(l) is torch.Tensor:
        acc += fn(l)
    elif type(l) is list:
        acc = sum([reduce_l(fn, sl, acc) for sl in l])
    else:
        raise Exception(f'type {type(l)} is not handled')
    return acc

def get_super_shape(si, sf):
    si, sf = list(si), list(sf)
    i, f = 0, 0
    super_shape = []
    while True:
        if i == len(si) or f == len(sf):
            assert i == len(si) and f == len(sf)
            return tuple(super_shape)
        if si[i] == sf[f]:
            super_shape.append(si[i])
            i += 1
            f += 1
        elif si[i] % sf[f] == 0:
            super_shape.append(sf[f])
            si[i] = si[i] // sf[f]
            f += 1
        elif sf[f] % si[i] == 0:
            super_shape.append(si[i])
            sf[f] = sf[f] // si[i]
            i += 1
        else:
            si[i+1] *= si[i]
            sf[f+1] *= sf[f]
            i += 1
            f += 1
        
def find_alignment(si, sf):
    si, sf = list(si), list(sf)
    i, f = 0, 0
    ass = []
    al = ([], [])
    while True:
        if i == len(si) or f == len(sf):
            assert i == len(si) and f == len(sf)
            return ass
        if si[i] == sf[f]:
            al[0].append(i)
            al[1].append(f)
            ass.append(al)
            al = ([], [])
            i += 1
            f += 1
        elif si[i] % sf[f] == 0:
            al[1].append(f)
            si[i] = si[i] // sf[f]
            f += 1
        elif sf[f] % si[i] == 0:
            al[0].append(i)
            sf[f] = sf[f] // si[i]
            i += 1
        else:
            return None
        
def flatten(l):
    if type(l) is list:
        lf = []
        for sl in l:
            lf.extend(flatten(sl))
        return lf
    else:
        return [l]

def remove_singleton(decomp):
    if type(decomp) is torch.Tensor:
        return decomp
    if len(decomp) == 1:
        return remove_singleton(decomp[0])
    else:
        return [remove_singleton(sub_decomp) for sub_decomp in decomp]

def decompose_transpose(root_node, root_val, root_shape, metadata):
    #print(f'{root_node} start')
    x, xi = root_node.next_functions[0]
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    #print(f'{root_node} decompose call')
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    #print(f'{root_node} decompose end')
    dim0 = convert_to_int(root_node._saved_dim0, len(x_shape))
    dim1 = convert_to_int(root_node._saved_dim1, len(x_shape))
    root_val = torch.transpose(x_val, dim0, dim1)
    root_decomp = map_l(lambda t: torch.transpose(t, dim0, dim1), x_decomp)
    #print(f'{root_node} end')
    #(debug) print(f'{root_node} error 2', (root_val - reduce_l(lambda t: t, root_decomp)).norm())
    return root_decomp, root_val

def decompose_t(root_node, root_val, root_shape, metadata):
    
    x, xi = root_node.next_functions[0]
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    root_val = x_val.T
    root_decomp = map_l(lambda t: t.T, x_decomp)
    return root_decomp, root_val

def decompose_roll(root_node, root_val, root_shape, metadata):
    
    x, xi = root_node.next_functions[0]
    shifts = root_node._saved_shifts
    shifts = [convert_to_int(shift, 0) for shift in shifts]
    dims = root_node._saved_dims
    dims = [convert_to_int(dim, len(root_shape)) for dim in dims]
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    # print(shifts, dims)
    root_val = torch.roll(x_val, shifts, dims)
    root_decomp = map_l(lambda t: torch.roll(t, shifts, dims), x_decomp)
    return root_decomp, root_val

def decompose_clone(root_node, root_val, root_shape, metadata):
    #print(f'{root_node} start')
    x, xi = root_node.next_functions[0]
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    #print(f'{root_node} decompose call')
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    #print(f'{root_node} decompose end')
    root_decomp, root_val = x_decomp, x_val
    #print(f'{root_node} end')
    #(debug) print(f'{root_node} error 2', (root_val - reduce_l(lambda t: t, root_decomp)).norm())
    return root_decomp, root_val

def decompose_permute(root_node, root_val, root_shape, metadata):
    #print(f'{root_node} start')
    perm_dim = root_node._saved_dims
    
    # print(perm_dim, root_node, root_val, root_shape)
    x, xi = root_node.next_functions[0]
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    
    perm_dim = [convert_to_int(dim, len(x_shape)) for dim in perm_dim]
    root_verify = x_val.permute(perm_dim)
    if root_val is None:
        root_val = root_verify
    else:
        assert (root_val - root_verify).norm() < 1e-6, f"Permute issue"
    
    root_decomp = map_l(lambda x: x.permute(perm_dim), x_decomp)
    #print(f'{root_node} end')
    #(debug) print(f'{root_node} error 2', (root_val - reduce_l(lambda t: t, root_decomp)).norm())
    return root_decomp, root_val

def decompose_slice(root_node, root_val, root_shape, metadata):
    #print(f'{root_node} start')
    dim, start, end = root_node._saved_dim, root_node._saved_start, root_node._saved_end
    
    x, xi = root_node.next_functions[0]
    
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    #print(f'{root_node} decompose call')
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    #print(f'{root_node} decompose end')
    end = min(end, x_shape[dim])
    dim = convert_to_int(dim, len(x_shape))
#     #(debug) print(x_val, dim, start, end-start)
    root_val = torch.narrow(x_val, dim, start, end-start)
    root_decomp = map_l(lambda t: torch.narrow(t, dim, start, end-start), x_decomp)
    #print(f'{root_node} end')
    #(debug) print(f'{root_node} error 2', (root_val - reduce_l(lambda t: t, root_decomp)).norm())
    return root_decomp, root_val

def decompose_expand(root_node, root_val, root_shape, metadata):
    #print(f'{root_node} start')
    x, xi = root_node.next_functions[0]
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    #print(f'{root_node} decompose call')
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    #print(f'{root_node} decompose end')
    root_val = x_val.expand(*root_shape)
    root_decomp = map_l(lambda t: t.expand(*root_shape), x_decomp)
    #print(f'{root_node} end')
    #(debug) print(f'{root_node} error 2', (root_val - reduce_l(lambda t: t, root_decomp)).norm())
    return root_decomp, root_val

def decompose_unbind(root_node, root_val, root_shape, index, metadata):
    #print(f'{root_node} start')
    x, xi = root_node.next_functions[0]
    
    try:
        root_node()
        nargs = 0
    except Exception as e:
        nargs = int(str(e).split(' ')[1])
        
    args = [None]*nargs
    args[index] = torch.zeros(*root_shape, device=metadata.device, dtype=metadata.dtype)
    x_shape, x_metadata = extract_shape_metadata(root_node(*args))
    del args
    #print(f'{root_node} decompose call')
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    #print(f'{root_node} decompose end')
    
    dim = convert_to_int(root_node._saved_dim, len(x_shape))
    root_val = x_val.unbind(dim=dim)[index]
    root_decomp = map_l(lambda t: t.unbind(dim=dim)[index], x_decomp)
    #print(f'{root_node} end')
    #(debug) print(f'{root_node} error 2', (root_val - reduce_l(lambda t: t, root_decomp)).norm())
    return root_decomp, root_val

def decompose_select(root_node, root_val, root_shape, metadata):
    dim, index = root_node._saved_dim, root_node._saved_index
    x, xi = root_node.next_functions[0]
    
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    
    dim = convert_to_int(dim, len(x_shape))
    root_val = torch.select(x_val, dim, index)
    root_decomp = map_l(lambda t: torch.select(t, dim, index), x_decomp)
    #(debug) print(f'{root_node} error 2', (root_val - reduce_l(lambda t: t, root_decomp)).norm())
    return root_decomp, root_val

def decompose_mul(root_node, root_val, root_shape, metadata):
    #print(f'{root_node} start')
    x, xi = root_node.next_functions[0]
    y, yi = root_node.next_functions[1]
    x_val, y_val = root_node._saved_self.data , root_node._saved_other.data 
    
    (x_shape, x_metadata), (y_shape, y_metadata) = [extract_shape_metadata(t) 
                                                    for t in root_node(torch.zeros(*root_shape, 
                                                                                 device=metadata.device, 
                                                                                 dtype=metadata.dtype))]
    
    x_decomp, x_val = decompose(x, x_val, x_shape, xi, x_metadata)
    y_decomp, y_val = decompose(y, y_val, y_shape, yi, y_metadata)
    
    if root_val is None:
        root_val = x_val * y_val
    else:
        pass
    
    device, dtype = metadata.device, metadata.dtype
    root_decomp = map_l(lambda t: map_l(lambda u: (u.to(device, dtype) * t.to(device, dtype)).cpu(), 
                                        x_decomp), 
                        y_decomp)

    return root_decomp, root_val

    
def decompose_div(root_node, root_val, root_shape, metadata):
    #print(f'{root_node} start')
    x, xi = root_node.next_functions[0]
    divisor = root_node._saved_other.data 
    
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype))[0])
    
    x_decomp, x_val = decompose(x, root_node._saved_self.data, x_shape, xi, x_metadata)
    
    if root_val is None:
        root_val = x_val / divisor
    else:
        pass
    
    device, dtype = metadata.device, metadata.dtype
    root_decomp = map_l(lambda t: (t.to(device, dtype)/divisor).cpu(), x_decomp)

    return root_decomp, root_val


def decompose_add(root_node, root_val, root_shape, metadata):
    #print(f'{root_node} start')
    x, xi = root_node.next_functions[0]
    y, yi = root_node.next_functions[1]
    x_val = getattr(root_node, '_saved_self', None)
    y_val = getattr(root_node, '_saved_other', None)
    
    (x_shape, x_metadata), (y_shape, y_metadata) = [extract_shape_metadata(t) 
                                                    for t in root_node(torch.zeros(*root_shape, 
                                                                                 device=metadata.device, 
                                                                                 dtype=metadata.dtype))]
    
    #print(f'{root_node} decompose call 1')
    x_decomp, x_val = decompose(x, x_val, x_shape, xi, x_metadata)
    #print(f'{root_node} decompose call 2')
    y_decomp, y_val = decompose(y, y_val, y_shape, yi, y_metadata)
    #print(f'{root_node} decompose end 2')
    
    if root_val is None:
        root_val = x_val + y_val
    else:
        pass
        #(debug) print(f'{root_node} error 1', (root_val - x_val - y_val).norm())
#         assert (root_val - x_val - y_val).norm() < 1e-6, f"Add issue {root_val.shape} {x_val.shape} {y_val.shape}"
        
    x_decomp = map_l(lambda t: t.expand(*root_shape), x_decomp)
    y_decomp = map_l(lambda t: t.expand(*root_shape), y_decomp)
    
    #(debug) print(f'Expansion {root_node}: 2')
    root_decomp = [x_decomp, y_decomp]
    #print(f'{root_node} end')
    #(debug) print(f'{root_node} error 2', (root_val - reduce_l(lambda t: t, root_decomp)).norm())
    return root_decomp, root_val
       
def decompose_reshape(root_node, root_val, root_shape, metadata):
    x, xi = root_node.next_functions[0]
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
        
    if x_val.shape != x_shape:
        error_dump.append(root_node)
        raise Exception()
    
    if root_val is None:
        root_val = x_val.reshape(*root_shape)
    
    root_decomp = map_l(lambda t: t.reshape(*root_shape), x_decomp)
    
    if 'expand' in root_node.metadata:
        super_shape = get_super_shape(root_shape, x_shape)
        align = find_alignment(super_shape, root_shape)
        # print('Expanding reshape', super_shape, root_shape, x_shape)

        many_to_one = [al for al in align if len(al[0]) > 1 and len(al[1]) == 1]
        many_to_one = sorted(many_to_one, key=lambda t: t[1][0], reverse=True)
        # print('Many to one', many_to_one)
        if many_to_one:
            for init_al, fin_al in many_to_one:
                fin_dim = fin_al[0]
                if fin_dim not in root_node.metadata['expand']:
                    continue
                chunk_size = super_shape[init_al[-1]]
                num_chunks = root_shape[fin_dim] // chunk_size
                assert len(fin_al) == 1
                root_decomp = map_l(partial(split, dim=fin_dim, chunk_size=chunk_size), 
                                    root_decomp, lambda x: x)
#                 print(f'Expansion {root_node}: {num_chunks}')
    
    #print(f'{root_node} end')
    #(debug) print(f'{root_node} error 2', (root_val - reduce_l(lambda t: t, root_decomp)).norm())
    return root_decomp, root_val

def decompose_layernorm(root_node, root_val, root_shape, metadata):
    '''
    Assumptions: 
    layer norm is only over last dim. 
    Only the input (x) is to be decomposed
    '''
    #print(f'{root_node} start')
    x, xi = root_node.next_functions[0]
    weight = root_node.next_functions[1][0].variable
    bias = root_node.next_functions[2][0].variable
    
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype))[0])
    
    #print(f'{root_node} decompose call')
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    #print(f'{root_node} decompose end')
    
    x_mean, x_std = x_val.mean(-1, keepdims=True), x_val.std(-1, keepdims=True)
    coeff = weight/(torch.sqrt(x_std**2 + 1e-5))
    const = bias - coeff*x_mean
    
    if root_val is None:
        root_val = coeff*x_val + const
    else:
        pass
        #(debug) print(f'{root_node} error 1', (root_val - coeff*x_val - const).norm())
        
    x_num_vecs = reduce_l(lambda t: 1, x_decomp)
    root_decomp = map_l(lambda t: (coeff*t.to(metadata.device, metadata.dtype) + (const/x_num_vecs)).cpu(), x_decomp)
    
    #print(f'{root_node} end')
    #(debug) print(f'{root_node} error 2', (root_val - reduce_l(lambda t: t, root_decomp)).norm())
    return root_decomp, root_val
    
def decompose_addmm(root_node, root_val, root_shape, metadata):
    ## y = Ax + b
    #print(f'{root_node} start')
    b = root_node.next_functions[0][0].variable.detach()
    A = root_node.next_functions[2][0].next_functions[0][0].variable.detach()
    x, xi = root_node.next_functions[1]
    x_val = root_node._saved_mat1
    if root_val is None:
        root_val = F.linear(x_val, A, bias=b)
    else:
        assert (root_val - F.linear(x_val, A, bias=b)).norm() < 1e-6, "ADDMM issue"

    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype))[1])
    x_decomp, _ = decompose(x, x_val, x_shape, xi, x_metadata)
    
    b = b/reduce_l(lambda x: 1, x_decomp)
    
    root_decomp = map_l(lambda t: F.linear(t.to(metadata.device, metadata.dtype), A, bias=b).cpu(), x_decomp)

    return root_decomp, root_val

def decompose_mm(root_node, root_val, root_shape, metadata):
    
    x, xi = root_node.next_functions[0]
    y, yi = root_node.next_functions[1]
    x_val = getattr(root_node, '_saved_self', None)
    y_val = getattr(root_node, '_saved_other', None)
    (x_shape, x_metadata), (y_shape, y_metadata) = [extract_shape_metadata(t) 
                                                    for t in root_node(torch.zeros(*root_shape, 
                                                                                 device=metadata.device, 
                                                                                 dtype=metadata.dtype))]    
    x_decomp, x_val = decompose(x, x_val, x_shape, xi, x_metadata)
    y_decomp, y_val = decompose(y, y_val, y_shape, yi, y_metadata)
    
    if root_val is None:
        root_val = x_val@y_val
    else:
        pass
#         assert (root_val - x_val@y_val).norm() < 1e-6, "Mm issue"
    device, dtype = metadata.device, metadata.dtype
    root_decomp = map_l(lambda t: map_l(lambda u: (u.to(device, dtype) @ t.to(device, dtype)).cpu(), 
                                        x_decomp), 
                        y_decomp)
    
    
    return root_decomp, root_val

def decompose_bmm(root_node, root_val, root_shape, metadata):
    (x, xi), (y, yi) = root_node.next_functions[0], root_node.next_functions[1]
    x_val, y_val = root_node._saved_self, root_node._saved_mat2
    (x_shape, x_metadata), (y_shape, y_metadata) = [extract_shape_metadata(t) 
                                                    for t in root_node(torch.zeros(*root_shape, 
                                                                                 device=metadata.device, 
                                                                                 dtype=metadata.dtype))] 
    
    #print(f'{root_node} decompose call 1')
    x_decomp, x_val = decompose(x, x_val, x_shape, xi, x_metadata)
    #print(f'{root_node} decompose call 2')
    y_decomp, y_val = decompose(y, y_val, y_shape, yi, y_metadata)
    #print(f'{root_node} decompose end')
    
    if root_val is None:
        root_val = torch.bmm(x_val, y_val)
    else:
        pass
        #(debug) print(f'{root_node} error 1', (root_val - torch.bmm(x_val, y_val)).norm())
        
    #(debug) print(x_val.shape, y_val.shape, reduce_l(lambda t: 1, x_decomp), reduce_l(lambda t: 1, y_decomp))
       
    device = metadata.device
    if 'expand' in root_node.metadata:
        # print('BMM expansion', x_val.shape, y_val.shape, root_shape)
        root_decomp = torch.einsum('n...ij,m...jk->nm...ikj', 
                                   torch.stack(flatten(x_decomp)).to(device), 
                                   torch.stack(flatten(y_decomp)).to(device)).cpu()
        root_decomp = list(torch.flatten(root_decomp, end_dim=1).unbind(0))
        # print('Expaned dims', len(root_decomp))
        root_decomp = map_l(lambda t: list(t.unbind(-1)), root_decomp)
    else:
        root_decomp = torch.einsum('n...ij,m...jk->nm...ik', 
                                   torch.stack(flatten(x_decomp)).to(device), 
                                   torch.stack(flatten(y_decomp)).to(device)).cpu()
        root_decomp = list(torch.flatten(root_decomp, end_dim=1).unbind(0))
    
    
    #(debug) print(f'Expansion {root_node}: {len(root_decomp[0])}')
    
#     root_decomp = map_l(lambda t: 
#                         map_l(lambda u: 
#                               list(torch.einsum('...ij,...jk->...ikj', u, t).unbind(-1)), 
#                               x_decomp, 
#                               lambda u: torch.bmm(u, t)), 
#                         y_decomp, 
#                         lambda t: map_l(lambda u: torch.bmm(u, t), x_decomp) )
    
    
    #print(f'{root_node} end')
    #(debug) print(f'{root_node} error 2', (root_val - reduce_l(lambda t: t, root_decomp)).norm())
    return root_decomp, root_val

def decompose_squeeze(root_node, root_val, root_shape, metadata):
    x, xi = root_node.next_functions[0]
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    dim = convert_to_int(root_node._saved_dim, len(x_shape))
    root_val = torch.squeeze(x_val, dim=dim)
    root_decomp = map_l(lambda t: torch.squeeze(t, dim=dim), x_decomp)
    return root_decomp, root_val

def decompose_unsqueeze(root_node, root_val, root_shape, metadata):
    x, xi = root_node.next_functions[0]
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    dim = convert_to_int(root_node._saved_dim, len(x_shape))
    root_val = torch.unsqueeze(x_val, dim=dim)
    root_decomp = map_l(lambda t: torch.unsqueeze(t, dim=dim), x_decomp)
    return root_decomp, root_val

def decompose_reductions(root_node, root_val, root_shape, metadata, typ='mean'):
    ## y = x.mean(dims)
    assert typ in ('mean', 'sum')
    
    x, xi = root_node.next_functions[0]
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    x_val = getattr(root_node, '_saved_self', None)

    x_decomp, x_val = decompose(x, x_val, x_shape, xi, x_metadata)
    
    red_dims = (convert_to_int(d, len(x_val.shape)) for d in root_node._saved_dim)
    red_dims = sorted(red_dims, reverse=True)
#     print('Reduction dims', red_dims)

    if typ == 'mean': 
        root_verify = x_val.mean(red_dims)
    elif typ == 'sum':
        root_verify = x_val.sum(red_dims)
        
    if root_val is None:
        root_val = root_verify
    else:
        err = (root_val - root_verify).norm()/root_val.norm()
        assert err < 1e-3, f"Reduction: {typ} issue, err: {err}"
    
    if 'expand' in root_node.metadata:
        root_decomp = x_decomp
        for dim in red_dims:
            scale = 1 if typ == 'sum' else x_val.shape[dim]
            root_decomp = map_l(lambda t: list((t/scale).unbind(dim)), root_decomp)
    else:
        if typ == 'mean': 
            root_decomp =  map_l(lambda t: t.mean(red_dims), x_decomp)
        elif typ == 'sum':
            root_decomp =  map_l(lambda t: t.sum(red_dims), x_decomp)
    
    return root_decomp, root_val

def decompose_conv(root_node, root_val, root_shape, metadata):
    x, xi = root_node.next_functions[0]
    weight = root_node.next_functions[1][0].variable
    bias = root_node.next_functions[2][0].variable
    stride, padding, dilation, groups = root_node._saved_stride, root_node._saved_padding, root_node._saved_dilation, root_node._saved_groups
    x_val = getattr(root_node, '_saved_self', None)
    
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype))[0])
    # print(root_node, x_shape, root_shape)
    x_decomp, x_val = decompose(x, x_val, x_shape, xi, x_metadata)
    
    if root_val is None:
        root_val = F.conv2d(x_val, weight, bias, stride, padding, dilation, groups)
    else:
        pass
        #(debug) print(f'{root_node} error 1', (root_val - F.conv2d(x_val, weight, bias, stride, padding, dilation, groups)).norm())
        
    device, dtype = metadata.device, metadata.dtype
    reduced_bias = bias/reduce_l(lambda t: 1, x_decomp)
    root_decomp = map_l(lambda t: F.conv2d(t.to(device, dtype), weight, reduced_bias, stride, padding, dilation, groups).cpu(), x_decomp)
    
    return root_decomp, root_val

def decompose_avgpool2d(root_node, root_val, root_shape, metadata):
    x, xi = root_node.next_functions[0]
    kernel_size, stride, padding, ceil_mode, count_include_pad = root_node._saved_kernel_size, root_node._saved_stride, root_node._saved_padding, root_node._saved_ceil_mode, root_node._saved_count_include_pad
    x_val = getattr(root_node, '_saved_self', None)
    
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    # print(root_node, x_shape, root_shape)
    x_decomp, x_val = decompose(x, x_val, x_shape, xi, x_metadata)
    
    if root_val is None:
        root_val = F.avg_pool2d(x_val, kernel_size, stride, padding, ceil_mode, count_include_pad)
    else:
        pass
        #(debug) print(f'{root_node} error 1', (root_val - F.avg_pool2d(x_val, kernel_size, stride, padding, ceil_mode, count_include_pad)).norm())
        
    device, dtype = metadata.device, metadata.dtype
    root_decomp = map_l(lambda t: F.avg_pool2d(t.to(device, dtype), kernel_size, stride, padding, ceil_mode, count_include_pad).cpu(), x_decomp)
    
    return root_decomp, root_val

def decompose_constant_pad(root_node, root_val, root_shape, metadata):
    x, xi = root_node.next_functions[0]
    pad = root_node._saved_pad
    
    x_shape, x_metadata = extract_shape_metadata(root_node(torch.zeros(*root_shape, 
                                                                        device=metadata.device, 
                                                                        dtype=metadata.dtype)))
    
    # print(root_node, x_shape, root_shape)
    x_decomp, x_val = decompose(x, None, x_shape, xi, x_metadata)
    
    if root_val is None:
        root_val = F.pad(x_val, pad)
    else:
        pass
        #(debug) print(f'{root_node} error 1', (root_val - F.pad(x_val, pad)).norm())
        
    device, dtype = metadata.device, metadata.dtype
    root_decomp = map_l(lambda t: F.pad(t.to(device, dtype), pad).cpu(), x_decomp)
    
    return root_decomp, root_val


def get_value(node, metadata):
    name = node.name()
    if name == 'torch::autograd::AccumulateGrad':
        return node.variable.detach()
    elif name[:-1] in ['ReluBackward', 'SoftmaxBackward', 'SafeSoftmaxBackward', 'SigmoidBackward']:
        return node._saved_result.detach()
    elif name[:-1] == 'GeluBackward':
        return F.gelu(node._saved_self, approximate=node._saved_approximate).detach()
    # elif name[:-1] == 'ConvolutionBackward':
    #     return F.conv2d(node._saved_input, node.next_functions[1][0].variable, node.next_functions[2][0].variable)
    else:
        error_dump.append(node)
        raise Exception(f'Type: {name} not handled')
    
def decompose(root_node, root_val, root_shape, index, metadata):
#     if root_node in store_decomp_dict and store_decomp_dict[root_node]:
#         return store_decomp_dict[root_node]
    name = root_node.name()[:-1]
#     print(f'Entering {root_node}')
    if name == 'AddmmBackward':
        root_decomp, root_val = decompose_addmm(root_node, root_val, root_shape, metadata)
    elif name == 'MmBackward':
        root_decomp, root_val = decompose_mm(root_node, root_val, root_shape, metadata)
    elif name == 'TBackward':
        root_decomp, root_val = decompose_t(root_node, root_val, root_shape, metadata)
    elif name == 'MeanBackward':
        root_decomp, root_val = decompose_reductions(root_node, root_val, root_shape, metadata, typ='mean')
    elif name == 'SumBackward':
        root_decomp, root_val = decompose_reductions(root_node, root_val, root_shape, metadata, typ='sum')
    elif name == 'PermuteBackward':
        root_decomp, root_val = decompose_permute(root_node, root_val, root_shape, metadata)
    elif name == 'ReshapeAliasBackward' or name == 'ViewBackward' or name == 'UnsafeViewBackward':
        root_decomp, root_val = decompose_reshape(root_node, root_val, root_shape, metadata)
    elif name == 'AddBackward':
        root_decomp, root_val = decompose_add(root_node, root_val, root_shape, metadata)
    elif name == 'SubBackward':
        root_decomp, root_val = decompose_sub(root_node, root_val, root_shape, metadata)
    elif name == 'MulBackward':
        root_decomp, root_val = decompose_mul(root_node, root_val, root_shape, metadata)
    elif name == 'DivBackward':
        root_decomp, root_val = decompose_div(root_node, root_val, root_shape, metadata)
    elif name == 'NativeLayerNormBackward':
        root_decomp, root_val = decompose_layernorm(root_node, root_val, root_shape, metadata)
    elif name == 'SelectBackward':
        root_decomp, root_val = decompose_select(root_node, root_val, root_shape, metadata)
    elif name == 'SliceBackward':
        root_decomp, root_val = decompose_slice(root_node, root_val, root_shape, metadata)
    elif name == 'CloneBackward' or name == 'ToCopyBackward':
        root_decomp, root_val = decompose_clone(root_node, root_val, root_shape, metadata)
    elif name == 'TransposeBackward':
        root_decomp, root_val = decompose_transpose(root_node, root_val, root_shape, metadata)
    elif name == 'ExpandBackward':
        root_decomp, root_val = decompose_expand(root_node, root_val, root_shape, metadata)
    elif name == 'UnbindBackward':
        root_decomp, root_val = decompose_unbind(root_node, root_val, root_shape, index, metadata)
    elif name == 'BmmBackward':
        root_decomp, root_val = decompose_bmm(root_node, root_val, root_shape, metadata) 
    elif name == 'SqueezeBackward':
        root_decomp, root_val = decompose_squeeze(root_node, root_val, root_shape, metadata) 
    elif name == 'UnsqueezeBackward':
        root_decomp, root_val = decompose_unsqueeze(root_node, root_val, root_shape, metadata) 
    elif name == 'RollBackward':
        root_decomp, root_val = decompose_roll(root_node, root_val, root_shape, metadata)
    elif name == 'ConvolutionBackward':
        root_decomp, root_val = decompose_conv(root_node, root_val, root_shape, metadata)
    elif name == 'AvgPool2DBackward':
        root_decomp, root_val = decompose_avgpool2d(root_node, root_val, root_shape, metadata)
    elif name == 'ConstantPadNdBackward':
        root_decomp, root_val = decompose_constant_pad(root_node, root_val, root_shape, metadata)
    else:
        #(debug) print(f'Terminating at {name}')
        root_val = get_value(root_node, metadata)
#         if getattr(root_val, is_input, False):
#             inputs.append(root_val)
        root_decomp = [root_val.cpu()]
#     print(f'Leaving {root_node}')
    return root_decomp, root_val

# def decompose_sub(root_node, root_val, gradient):
    
#     x, xi = root_node.next_functions[0]
#     y, yi = root_node.next_functions[1]
#     x_val = getattr(root_node, '_saved_self', None)
#     y_val = getattr(root_node, '_saved_other', None)
#     x_grad, y_grad = root_node(gradient)
    
#     x_decomp, x_val = decompose(x, x_val, x_grad, xi, metadata)
#     y_decomp, y_val = decompose(y, y_val, y_grad, yi, metadata)
    
#     if root_val is None:
#         root_val = x_val - y_val
#     else:
#         assert (root_val - x_val + y_val).norm() < 1e-6, "Sub issue"
        
#     x_decomp = map_l(lambda t: t.expand(*gradient.shape), x_decomp)
#     y_decomp = map_l(lambda t: t.expand(*gradient.shape), y_decomp)
#     root_decomp = [x_decomp, map_l(lambda t: -t, y_decomp)]
#     return root_decomp, root_val
 
