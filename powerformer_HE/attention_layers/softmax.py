from base_fncs import *
import numpy as np

def precompute_softmax(rd) :
    rd = np.transpose(rd, (1, 0, 2))
    rd1 = rd[:,:,:64].reshape(128,-1)#real
    rd2 = rd[:,:,64:].reshape(128,-1)#cplx
    rd1 = [bias_pack(rd1[:,i*256:(i+1)*256]) for i in range(3)]
    rd2 = [bias_pack(rd2[:,i*256:(i+1)*256]) for i in range(3)]
    return np.array(rd1+rd2,dtype=np.complex64)

def compute_mask(mask) :
    m1 = mask.reshape(1,-1)[:,:64]
    m2 = mask.reshape(1,-1)[:,64:]
    m1 = np.tile(m1,(128,4))
    m2 = np.tile(m2,(128,4))
    m1 = bias_pack(m1)
    m2 = bias_pack(m2)
    return m1,m2

def apply_softmax(x,precompute,mask) :
    rd = precompute
    m1,m2 = compute_mask(mask)
    x = [add(data,5) for data in x]
    x1 = [cc_mult(data,data) for data in x]
    x1 = [cc_mult(data,data) for data in x1]
    l = len(x)//2
    for i in range(l) :
        x[i] = add(cc_mult(x1[i],cp_mult(x[i],m1*rd[i],rs=True,pad=False)),mult_i(cc_mult(x1[i+l],cp_mult(x[i+l],m2*rd[i+l],rs=True,pad=False))))
    x = x[:l]
    return x
