from base_fncs import *
import numpy as np

def make_transpose_mask(shape,d,rd1,rd2,n) :
    m = [np.zeros((d,d)) for _ in range(d)]
    m2 = [np.zeros((d,d)) for _ in range(d)]
    ident = np.eye(d)
    for i in range(d) :
        m[i] += np.roll(ident,-i,axis=1)
        m[i][:i] = 0
        m2[d-i-1] = m[i].T
        m[i] = np.tile(m[i],(shape[0]//d, shape[1]//d))
        m2[d-i-1] = np.tile(m2[d-i-1],(shape[0]//d, shape[1]//d))
    
    m = m + m2[:-1]
    chgm = [np.zeros_like(m[0]) for _ in range(len(m))]
    for k in range(len(m)) :
        for j in range(64) :
            for i in range(4) :
                chgm[k][j*4+i,:] = m[k][i*64+j,:]
        chgm[k] = p_column_wise_pack(chgm[k].T)
        
    
    for j in range(rd2) :
        for i in range(-rd1,rd1) :
            now = rd2*i+j
            if abs(now) >= d : continue
            chgm[now] = p_rot(chgm[now],j*(shape[1]*n-1))

    return chgm

def apply_transpose(a,shape,d,rd1,rd2,masks,n=1) :
    if shape[0]%d !=0 or shape[1]%d != 0 or rd1*rd2 < d : 
        print('shape error')
        return
    
    bs = [None for _ in range(rd1*2)]
    rj = None
    for j in range(rd2) :
        if j == 0 : rj = a
        else : rj = rot(rj,(shape[1]*n-1))
        for i in range(-rd1,rd1) :
            now = rd2*i+j
            if abs(now) >= d : continue
            cm = cp_mult(rj,[masks[now]],False)
            bs[i] = add(bs[i],cm) if bs[i] is not None else cm
            
    tot = None
    for i in range(-rd1,rd1) :
        if bs[i] is None : continue
        next = i*rd2*(shape[1]*n-1)
        bs[i] = rescale(bs[i])
        tot = add(tot,rot(bs[i],next)) if tot is not None else rot(bs[i],next)
    return tot