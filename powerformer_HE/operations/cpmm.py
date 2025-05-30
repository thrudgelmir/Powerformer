from base_fncs import *
import numpy as np

def cpmm_plain_pack(b,shapes) :
    shape,st_shape,mid_shape,ed_shape,d,rd1,rd2,plain_num = shapes
    ident = np.eye(d)
    row_chg = np.zeros_like(b)
    col_chg = np.zeros_like(b)
    for k in range(mid_shape//d) :
        for j in range(64) :
            for i in range(4) :
                row_chg[k*256+j*4+i,:] = b[k*256+i*64+j,:]
    for k in range(ed_shape//d) :
        for j in range(64) :
            for i in range(4) :
                col_chg[:,k*256+j*4+i] = row_chg[:,k*256+i*64+j]


    idents = np.array([np.roll(ident,i,axis=0) for i in range(d)])
    div_b = np.array([[col_chg[d*m:d*(m+1),d*e:d*(e+1)] for e in range(ed_shape//d)] for m in range(mid_shape//d)],dtype=np.complex64)
    div_b = np.array([div_b*ident for ident in idents])
    plain = np.repeat(np.sum(div_b,axis=-2,keepdims=True),st_shape,axis=-2)
    p = [[[p_column_wise_pack(plain[i,m,e,:])for e in range(ed_shape//d)] for m in range(mid_shape//d)] for i in range(d)]
    for j in range(rd2) :
        for m in range(mid_shape//d) :
            for i in range(rd1) :
                now = rd2*i+j
                next = i*rd2*st_shape
                if abs(now) >= d : continue
                for e in range(ed_shape//d) :
                    p[now][m][e] = p_rot(p[now][m][e],-next)
                    
    return np.array(p)


def apply_cpmm(a, b, shapes, qkv=False) :
    shape,st_shape,mid_shape,ed_shape,d,rd1,rd2,plain_num = shapes
    bs = [[[None for _ in range(rd1)]for _ in range(ed_shape//d)] for _ in range(plain_num)]
    rj = None
    for m in range(mid_shape//d) :
        for j in range(rd2) :
            if j == 0 : rj = clone(a[m])
            else : rj = rot(rj,st_shape)
            for i in range(rd1) :
                now = rd2*i+j
                next = i*rd2*st_shape
                if abs(now) >= d : continue
                for e in range(ed_shape//d) :
                    for l in range(plain_num) :
                        if l==0 and e==2 and qkv: continue
                        cm = cp_mult(rj,[b[l][now][m][e]],False,True)
                        bs[l][e][i] = add(bs[l][e][i],cm) if bs[l][e][i] is not None else cm #j aggregate

    tot = [[None for _ in range(ed_shape//d)] for _ in range(plain_num)]
    for i in range(rd1) :
        next = i*rd2*st_shape
        for l in range(plain_num) :
            for e in range(ed_shape//d) :
                if bs[l][e][i] is None : continue
                bs[l][e][i] = rescale(bs[l][e][i])
                tot[l][e] = add(tot[l][e],rot(bs[l][e][i],next)) if tot[l][e] is not None else rot(bs[l][e][i],next)

    return tot
