from base_fncs import *
import numpy as np
from operations.cpmm import cpmm_plain_pack, apply_cpmm
import math

def show(name,data=cnts) :
    form = "    ".join(f"{k}: {v:<5}" for k, v in data.items())
    print(f"{name}\t{form}")
    add_tot()
    clear()

def precompute_layer2(w,b) :
    st_shape = 128
    mid_shape = 3072//2
    ed_shape = 768
    plain_num = 1

    d = s//st_shape
    k = mid_shape/ed_shape/plain_num
    if k > 1 : 
        rd2 = round(np.sqrt(d/k))
        rd1 = math.ceil(d/rd2)
    else : 
        rd1 = round(np.sqrt(d*k))
        rd2 = math.ceil(d/rd1)
    complexw = [w[0][:3072//2,:]/2 + w[0][3072//2:,:] * -1j/2]
    shape = (d,st_shape)
    shapes = (shape,st_shape,mid_shape,ed_shape,d,rd1,rd2,plain_num)
    complexw = [cpmm_plain_pack(data,shapes) for data in complexw]
    
    complexw = pre_pack(complexw,24)

    b = np.array([bias_pack(b[:,i*256:(i+1)*256])for i in range(3)])
    return (complexw,b,shapes)

def apply_layer2(a,precomputes,verbose) :

    w,b,shapes = precomputes
    
    if verbose : clear_tot(); clear()
    l = len(a)//2
    for i in range(len(a)//2) :
        a[i] = add(a[i],mult_i(a[i+l]))
    calcs = apply_cpmm(a,w,shapes)[0]
    if verbose : show('cpmm')

    conj = [conjugate(data) for data in calcs]
    res = [add(d1,d2) for d1,d2 in zip(calcs,conj)]
    res = [add(d1,d2) for d1,d2 in zip(res,b)]
    if verbose : show('conj')
    if verbose : print('*'*38,'ffn layer2 tot','*'*38); show('tot ',tot_cnts); print('*'*92)
    return res