from base_fncs import *
import numpy as np

def precompute_layer1(w,b) :
    divw = [w[256*i:256*(i+1),:]for i in range(3)]
    divw = [[data[:,128*i:128*(i+1)]for i in range(6)]for data in divw]
    divb = [b[:,128*i:128*(i+1)]for i in range(6)]
    
    for i in range(len(divb)) :
        zero = np.zeros((256,128))
        zero[:1,:] = divb[i]
        divb[i] = zero

    chgw = [[np.zeros_like(divw[0][0]) for _ in range(len(divw[0]))] for _ in range(len(divw))]
    chgb = [np.zeros_like(divb[0]) for _ in range(len(divb))]

    for k in range(len(chgw)) :
        for l in range(len(chgw[k])) :
            for j in range(64) :
                for i in range(4) :
                    chgw[k][l][j*4+i,:] = divw[k][l][i*64+j,:]
            chgw[k][l] = p_column_wise_pack(chgw[k][l].T)
        
    for k in range(len(divb)) :
        for j in range(64) :
            for i in range(4) :
                chgb[k][j*4+i,:] = divb[k][i*64+j,:]
        chgb[k] = p_column_wise_pack(chgb[k].T)

    mask = np.zeros((256,128))
    mask[:,:1] = 1
    mask = p_column_wise_pack(mask.T)

    chgw = pre_pack(chgw,17)
    mask = pre_pack([mask],16)[0]
    return chgw,chgb,mask


def apply_layer1(a,precomputes,verbose) :

    w,b,mask = precomputes
    
    if verbose : clear_tot(); clear()
    
    a = [cp_mult(data,[mask]) for data in a]
    for k in range(3) : 
        for i in range(7) :
            a[k] = add(a[k],rot(a[k],-2**i))

    tot = [None for _ in range(6)]
    for i in range(3) :
        for j in range(6) :
            tot[j] = cp_mult(a[i],[w[i][j]]) if tot[j] is None else add(tot[j],cp_mult(a[i],[w[i][j]]))
    
    
    for k in range(6) :
        for i in range(8) :
            tot[k] = add(tot[k],rot(tot[k],2**i*128))

    for k in range(6) :
        tot[k] = add(tot[k],b[k])

    if verbose : print(cnts); add_tot(); clear(); print('*'*20,'layer1 tot','*'*20); print(tot_cnts); print('*'*52)
    return tot