from base_fncs import *
import numpy as np
import copy

def make_tau_mask(shape,d,rd1,rd2,n=1) :
    m = [[] for _ in range(3)]
    m[0] = [np.zeros((d,d)) for _ in range(d)]
    for i in range(d) :
        m[0][i][:(d-i),i] = 1
        m[0][i] = np.tile(m[0][i],(shape[0]//d, shape[1]//d))
        
    m[1] = [np.zeros((d,d)) for _ in range(d)]
    for i in range(d) :
        m[1][i][(d-i):d,i] = 1
        m[1][i] = np.tile(m[1][i],(shape[0]//d, shape[1]//d))
    m[2] = [np.zeros((d,d)) for _ in range(d)]
    for i in range(d) :
        m[2][i][:,i] = 1
        m[2][i] = np.tile(m[2][i],(shape[0]//d, shape[1]//d))

    chgm = copy.deepcopy(m)
    for c in range(3):
        for k in range(len(m[c])) :
            for j in range(64) :
                for i in range(4) :
                    chgm[c][k][j*4+i,:] = m[c][k][i*64+j,:]
            chgm[c][k] = p_column_wise_pack(chgm[c][k].T)
    
    for j in range(rd2) :
        for i in range(rd1) :
            now = rd2*i+j
            next = i*rd2*shape[1]*n
            if abs(now) >= d : continue
            for k in range(3) :
                chgm[k][now] = p_rot(chgm[k][now],-next)

    return chgm

def apply_tau(a,shape,d,rd1,rd2,masks,n=1) :
    bs = [None for _ in range(rd1)]
    rj = None
    for j in range(rd2) :
        if j == 0 : rj = a
        else : rj = rot(rj,shape[1]*n)
        for i in range(rd1) :
            now = rd2*i+j
            next = i*rd2*shape[1]*n
            if abs(now) >= d : continue
            cm = cp_mult(rj,[masks[2][now]],False)
            bs[i] = add(bs[i],cm) if bs[i] is not None else cm
            
    tot = None
    for i in range(rd1) :
        if bs[i] is None : continue
        bs[i] = rescale(bs[i])
        next = i*rd2*shape[1]*n
        tot = add(tot,rot(bs[i],next)) if tot is not None else rot(bs[i],next)
    return tot


def make_sigma_mask(shape,d,rd1,rd2) :
    m = [[] for _ in range(2)]
    m[0] = [np.zeros((d,d)) for _ in range(d)]
    m[1] = [np.zeros((d,d)) for _ in range(d)]
    for i in range(d) :
        m[0][i][i,:d-i] = 1
        m[0][i] = np.tile(m[0][i],(shape[0]//d, shape[1]//d))
        m[1][i][i,d-i:] = 1
        m[1][i] = np.tile(m[1][i],(shape[0]//d, shape[1]//d))

    chgm = copy.deepcopy(m)
    for c in range(2):
        for k in range(len(m[c])) :
            for j in range(64) :
                for i in range(4) :
                    chgm[c][k][j*4+i,:] = m[c][k][i*64+j,:]
            chgm[c][k] = p_column_wise_pack(chgm[c][k].T)
    
    for j in range(rd2) :
        for i in range(rd1) :
            now = rd2*i+j
            next = i*rd2
            if abs(now) >= d : continue
            for c in range(2) :
                chgm[c][now] = p_rot(chgm[c][now],-next)
    return chgm

def apply_sigma(a, shape, d, rd1, rd2, masks) :
    bs = [None for _ in range(rd1)]
    rj,rj2 = None,None
    for j in range(rd2) :
        if j == 0 : 
            rj = a
            rj2 = rot(a,-d)
        else :
            rj = rot(rj,1)
            rj2 = rot(rj2,1)
        for i in range(rd1) :
            now = rd2*i+j
            if abs(now) >= d : continue
            cm = add(cp_mult(rj,[masks[0][now]],False),cp_mult(rj2,[masks[1][now]],False))
            bs[i] = add(bs[i],cm) if bs[i] is not None else cm
    tot = None
    for i in range(rd1) :
        if bs[i] is None : continue
        next = i*rd2
        bs[i] = rescale(bs[i])
        tot = add(tot,rot(bs[i],next)) if tot is not None else rot(bs[i],next)
    return tot

def make_ccmm_mask(shape,d,rd1,rd2) :
    phi_m = [np.zeros((d,d)) for _ in range(d)]
    for i in range(d) :
        phi_m[i][:,i:] = 1
        phi_m[i] = np.tile(phi_m[i],(shape[0]//d, shape[1]//d))
    chgm = copy.deepcopy(phi_m)
    for k in range(len(phi_m)) :
        for j in range(64) :
            for i in range(4) :
                chgm[k][j*4+i,:] = phi_m[k][i*64+j,:]
        chgm[k] = p_column_wise_pack(chgm[k].T)
    chgm = np.array(chgm,dtype=np.complex64)
    return np.array([chgm,1-chgm])

def apply_ccmm(a, b, shape, d, rd1, rd2, masks, n=1) :
    bs = [None for _ in range(rd1)]
    pre_phi = rot(a,-d)
    rj = None
    for j in range(rd2) :
        if j == 0 : rj = b
        else : rj = rot(rj,shape[1]*n)
        for i in range(rd1) :
            now = rd2*i+j
            next = i*rd2*shape[1]*n
            if abs(now) >= d : continue
            psi = rj
            phi = rot(rescale(add(cp_mult(a,[masks[0][now]],False),cp_mult(pre_phi,[masks[1][now]],False))),now-next)
            cm = cc_mult(psi,phi,False)
            bs[i] = add(bs[i],cm) if bs[i] is not None else cm

    tot = None
    for i in range(rd1) :
        if bs[i] is None : continue
        bs[i] = ks(bs[i])
        next = i*rd2*shape[1]*n
        tot = add(tot,rot(bs[i],next)) if tot is not None else rot(bs[i],next)

    return tot