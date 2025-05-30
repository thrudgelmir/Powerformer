from base_fncs import *
import numpy as np
from operations.cpmm import cpmm_plain_pack, apply_cpmm
import math

def show(name,data=cnts) :
    form = "    ".join(f"{k}: {v:<5}" for k, v in data.items())
    print(f"{name}\t{form}")
    add_tot()
    clear()

def precompute_layer1(w,b) :
    st_shape = 128
    mid_shape = 768
    ed_shape = 3072//2
    plain_num = 1

    d = s//st_shape
    k = mid_shape/ed_shape/plain_num
    if k > 1 : 
        rd2 = round(np.sqrt(d/k))
        rd1 = math.ceil(d/rd2)
    else : 
        rd1 = round(np.sqrt(d*k))
        rd2 = math.ceil(d/rd1)
    complexw = [w[0][:,:3072//2]/2 + w[0][:,3072//2:] * -1j/2]
    shape = (d,st_shape)
    shapes = (shape,st_shape,mid_shape,ed_shape,d,rd1,rd2,plain_num)
    complexw = [cpmm_plain_pack(data,shapes)/80 for data in complexw]
    
    b = np.array([bias_pack(b[:,i*256:(i+1)*256]) for i in range(12)])/80

    complexw = pre_pack(complexw,19)
    return (complexw,b,shapes)


def apply_layer1(a,precomputes,verbose) :
    w,b,shapes = precomputes
    
    if verbose : clear_tot(); clear()
    
    calcs = apply_cpmm(a,w,shapes)[0]

    if verbose : show('cpmm')
    l = len(calcs)
    for i in range(l):
        conj = conjugate(calcs[i])
        real = add(calcs[i],conj)
        cplx = mult_i(sub(calcs[i],conj))
        calcs[i] = real
        calcs.append(cplx)

    calcs = [add(d1,d2) for d1,d2 in zip(calcs,b)]
    if verbose : show('conj')
    
    if verbose : print('*'*38,'ffn layer1 tot','*'*38); show('tot ',tot_cnts); print('*'*92)
    return calcs