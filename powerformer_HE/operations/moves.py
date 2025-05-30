from base_fncs import *
import numpy as np

def make_move1_mask(shape) :
    mask = np.zeros(shape).T
    mask[:shape[1]//2,:] = 1
    mask2 = 1-mask
    mask = p_column_wise_pack(mask)
    mask2 = p_column_wise_pack(mask2)
    return [mask,mask2]


def apply_move1(a, shape, masks) :
    
    half1 = cp_mult(a,[masks[0]])
    half2 = sub(a,half1)
    half1 = add(half1,rot(half1,-shape[1]//2))
    half2 = add(half2,rot(half2,shape[1]//2))
    return [half1,half2]