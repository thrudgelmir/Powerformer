from base_fncs import *
import numpy as np
from operations.transpose import make_transpose_mask,apply_transpose
from operations.moves import make_move1_mask, apply_move1
from operations.ccmm import make_ccmm_mask, make_sigma_mask, make_tau_mask
from operations.ccmm import apply_ccmm, apply_sigma, apply_tau

def show(name,data=cnts) :
    form = "    ".join(f"{k}: {v:<5}" for k, v in data.items())
    print(f"{name}\t{form}")
    add_tot()
    clear()

def precompute_layer2(shape):
    d,rd1,rd2 = 64,5,13
    transpose_mask = make_transpose_mask(shape,d,rd1,rd2,4)
    transpose_mask = pre_pack(transpose_mask,16)

    move1_mask = make_move1_mask(shape)
    move1_mask = pre_pack(move1_mask,18)

    d,rd1,rd2 = 64,13,5
    sigma_mask = make_sigma_mask(shape,d,rd1,rd2)
    sigma_mask = pre_pack(sigma_mask,17)

    d,rd1,rd2 = 64,8,8
    tau_mask = make_tau_mask(shape,d,rd1,rd2,4)
    tau_mask2 = np.array(tau_mask)/4
    tau_mask = pre_pack(tau_mask,16)
    tau_mask2 = pre_pack(tau_mask2,16)

    ccmm_mask = make_ccmm_mask(shape,d,rd1,rd2)
    ccmm_mask = pre_pack(ccmm_mask,19)

    return (transpose_mask,move1_mask,sigma_mask,tau_mask,tau_mask2,ccmm_mask)

def apply_layer2(q,k,shape,precomputes,verbose) :
    transpose_mask,move1_mask,sigma_mask,tau_mask,tau_mask2,ccmm_mask = precomputes

    d,rd1,rd2 = 64,5,13
                        
    if verbose : clear_tot(); clear()
    k[1] = add(k[1],mult_i(k[2]))
    k = k[:2]
    k = [apply_transpose(cipher,shape,d,rd1,rd2,transpose_mask,4) for cipher in k]
    if verbose : show('trans')

    d,rd1,rd2 = 64,13,5
    k = [apply_sigma(cipher,shape,d,rd1,rd2,sigma_mask)for cipher in k]
    if verbose : show('sigma')

    k = [apply_move1(cipher,shape,move1_mask) for cipher in k]
    if verbose : show('move')

    k.append([None for _ in range(len(k[1]))])
    for i in range(len(k[1])) :
        conj = conjugate(k[1][i])
        k[2][i] = mult_i(sub(k[1][i],conj))
        k[1][i] = add(k[1][i],conj)
        
    for i in range(len(k)) :
        k[i][0] = add(k[i][0],mult_i(k[i][1]))
        k[i] = k[i][0]

    q[1] = add(q[1],mult_i(q[2]))
    q = q[:2]

    if verbose : show('conj')

    d,rd1,rd2 = 64,8,8
    for i in range(len(q)) :
        if i == 0 : q[i] = apply_tau(q[i],shape,d,rd1,rd2,tau_mask,4)
        else : q[i] = apply_tau(q[i],shape,d,rd1,rd2,tau_mask2,4)

    if verbose : show('tau ')

    conj = conjugate(q[1])
    q.append(mult_i(sub(q[1],conj)))
    q[1] = add(q[1],conj)

    if verbose : show('conj')

    res = []
    for i in range(len(k)) :
        res.append(apply_ccmm(k[i],q[i],shape,d,rd1,rd2,ccmm_mask,4))

    if verbose : show('ccmm')

    conj = [conjugate(data) for data in res]
    real = [add(d1,d2) for d1,d2 in zip(res,conj)]
    cplx = [mult_i(sub(d2,d1)) for d1,d2 in zip(res,conj)]

    if verbose : show('conj')

    if verbose : print('*'*35,'attention layer2 tot','*'*35); show('tot ',tot_cnts); print('*'*92)

    return real+cplx