from base_fncs import *
import numpy as np
from operations.cpmm import cpmm_plain_pack, apply_cpmm
import math

def show(name,data=cnts) :
    form = "    ".join(f"{k}: {v:<5}" for k, v in data.items())
    print(f"{name}\t{form}")
    add_tot()
    clear()

def precompute_layer1(w,bq,bk,bv) :
    st_shape = 128
    mid_shape = 768
    ed_shape = 768
    plain_num = 2
    d = s//st_shape
    k = mid_shape/5/d
    if k > 1 : 
        rd2 = round(np.sqrt(d/k))
        rd1 = math.ceil(d/rd2)
    else : 
        rd1 = round(np.sqrt(d*k))
        rd2 = math.ceil(d/rd1)
    shape = (d,st_shape)
    shapes = (shape,st_shape,mid_shape,ed_shape,d,rd1,rd2,plain_num)

    packedw = [cpmm_plain_pack(data,shapes) for data in w]
    packedw[0][:,:,1,:] += packedw[0][:,:,2,:] * -1j
    packedw[0][:,:,1,:] /= 2
    packedw[0] = packedw[0][:,:,:2,:]/2
    packedw[1] /= 16 
    packedw[2] *= -1j/2
    
    packedw[1] += packedw[2]
    complexw = packedw[:2]

    complexw[0] = pre_pack(complexw[0],15)
    complexw[1] = pre_pack(complexw[1],15)

    bq = [bias_pack(bq[:,i*256:(i+1)*256])/2 for i in range(3)]
    bk = [bias_pack(bk[:,i*256:(i+1)*256])/8 for i in range(3)]
    bv = [bias_pack(bv[:,i*256:(i+1)*256]) for i in range(3)]

    return [complexw,bq,bk,bv,shapes]


def apply_layer1(a,precomputes,verbose) :
    w,bq,bk,bv,shapes = precomputes
    
    if verbose : clear_tot(); clear()
    
    q,kv = apply_cpmm(a,w,shapes,True)
    if verbose : show('cpmm')
    cg = [conjugate(data) for data in kv]
    k = [add(d1,d2)for d1,d2 in zip(kv,cg)]
    v = [mult_i(sub(d1,d2))for d1,d2 in zip(kv,cg)]
    cg = conjugate(q[1])
    q[2] = mult_i(sub(q[1],cg))
    q[1] = add(q[1],cg)
    
    q = [add(d1,d2) for d1,d2 in zip(q,bq)]
    k = [add(d1,d2) for d1,d2 in zip(k,bk)]
    v = [add(d1,d2) for d1,d2 in zip(v,bv)]
    if verbose : show('conj')
    if verbose : print('*'*35,'attention layer1 tot','*'*35); show('tot ',tot_cnts); print('*'*92)
    return q,k,v