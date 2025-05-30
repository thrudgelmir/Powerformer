from base_fncs import *
import numpy as np
from operations.moves import make_move1_mask, apply_move1
from operations.ccmm import make_ccmm_mask, make_sigma_mask, make_tau_mask
from operations.ccmm import apply_ccmm, apply_sigma, apply_tau
import time

def show(name,data=cnts) :
    form = "    ".join(f"{k}: {v:<5}" for k, v in data.items())
    print(f"{name}\t{form}")
    add_tot()
    clear()
    
def precompute_layer3(shape, bts_range):
    d,rd1,rd2 = 64,5,13
    move1_mask = make_move1_mask(shape)
    move1_mask = pre_pack(move1_mask,17)

    d,rd1,rd2 = 64,13,5
    sigma_mask = make_sigma_mask(shape,d,rd1,rd2)
    sigma_mask = pre_pack(sigma_mask,16)


    d,rd1,rd2 = 64,8,8
    tau_mask = make_tau_mask(shape,d,rd1,rd2,4)
    tau_mask = np.array(tau_mask)/2
    tau_mask2 = [data/2 for data in tau_mask]
    tau_mask = pre_pack(tau_mask,24)
    tau_mask2 = pre_pack(tau_mask2,24)

    ccmm_mask = make_ccmm_mask(shape,d,rd1,rd2)
    ccmm_mask = pre_pack(ccmm_mask/bts_range,18)
    return [move1_mask,sigma_mask,tau_mask,tau_mask2,ccmm_mask]

def apply_layer3(a,v,shape,precomputes,bts_range, verbose) :
    move1_mask,sigma_mask,tau_mask,tau_mask2,ccmm_mask = precomputes

    d,rd1,rd2 = 64,13,5

    if verbose : clear_tot(); clear()

    v[1] = add(v[1],mult_i(v[2]))
    v = v[:2]
    v = [apply_sigma(cipher,shape,d,rd1,rd2,sigma_mask)for cipher in v]    
    if verbose : show('sigma')

    v = [apply_move1(cipher,shape,move1_mask) for cipher in v]
    if verbose : show('move')

    v.append([None for _ in range(len(v[1]))])
    for i in range(len(v[1])) :
        conj = conjugate(v[1][i])
        v[2][i] = mult_i(sub(conj,v[1][i]))
        v[1][i] = add(v[1][i],conj)
            
    for i in range(len(v)) :
        v[i][0] = sub(v[i][0],mult_i(v[i][1]))
        v[i] = v[i][0]

    d,rd1,rd2 = 64,8,8
    if verbose : show('conj')

    for i in range(len(a)) :
        if i == 0 : a[i] = apply_tau(a[i],shape,d,rd1,rd2,tau_mask,4)
        else : a[i] = apply_tau(a[i],shape,d,rd1,rd2,tau_mask2,4)
    if verbose : show('tau ')

    res = [apply_ccmm(v[i],a[i],shape,d,rd1,rd2,ccmm_mask,4) for i in range(len(a))]
    if verbose : show('ccmm')

    for i in range(len(res)) :
        res[i] = add(res[i],conjugate(res[i]))
        
    if verbose : show('conj')
    res[1] = sub(res[1],mult_i(res[2]))
    res = res[:2]
    st = time.time()
    res = [bootstrap(data) for data in res]
    ed = time.time()
    conj = conjugate(res[1])
    real = add(res[1],conj)
    cplx = mult_i(sub(res[1],conj))
    res[1] = real
    res.append(cplx)    
    res[0] = cp_mult(res[0],bts_range)
    res[1] = cp_mult(res[1],bts_range//2)
    res[2] = cp_mult(res[2],bts_range//2)
    
    #for gpu options       
    #l = res[0].level
    #res = [set_level(data,l+2) for data in res]

    if verbose : print('*'*35,'attention layer3 tot','*'*35); show('tot ',tot_cnts); print('*'*92)
    return res, ed-st
