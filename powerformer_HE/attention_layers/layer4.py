from base_fncs import *
import numpy as np
from operations.cpmm import cpmm_plain_pack, apply_cpmm
import math

def show(name,data=cnts) :
    form = "    ".join(f"{k}: {v:<5}" for k, v in data.items())
    print(f"{name}\t{form}")
    add_tot()
    clear()
    
def precompute_layer4(w,b) :
    st_shape = 128
    mid_shape = 768
    ed_shape = 768
    plain_num = 1

    d = s//st_shape
    k = mid_shape/(ed_shape*plain_num/3*2)

    if k > 1 : 
        rd2 = round(np.sqrt(d/k))
        rd1 = math.ceil(d/rd2)
    else : 
        rd1 = round(np.sqrt(d*k))
        rd2 = math.ceil(d/rd1)

    shape = (d,st_shape)
    shapes = (shape,st_shape,mid_shape,ed_shape,d,rd1,rd2,plain_num)
    packedw = cpmm_plain_pack(w[0],shapes)

    ed_shape = 768//3*2
    shapes = (shape,st_shape,mid_shape,ed_shape,d,rd1,rd2,plain_num)

    packedw[:,:,1] += packedw[:,:,2] * -1j
    packedw[:,:,1] /= 2
    packedw = packedw[:,:,:2]
    packedw = pre_pack(packedw,17)
    b = [bias_pack(b[:,i*256:(i+1)*256]) for i in range(3)]
    
    return [[packedw], b, shapes]


def apply_layer4(a,precomputes,verbose) :
    w,b,shapes= precomputes
    if verbose : clear_tot(); clear()
    res = apply_cpmm(a,w,shapes)[0]
    if verbose : show('cpmm')
    conj = conjugate(res[1])
    res.append(mult_i(sub(res[1],conj)))
    res[1] = add(res[1],conj)

    res = [add(d1,d2) for d1,d2 in zip(res,b)]
    if verbose : show('conj')
    if verbose : print('*'*35,'attention layer4 tot','*'*35); show('tot ',tot_cnts); print('*'*92)
    return res