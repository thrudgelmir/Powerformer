from base_fncs import *

def apply_gelu(x,data) :
    ret = cp_mult(x,data[1])
    x2 = cc_mult(x,x)
    ret = add(ret,cc_mult(x2,cp_mult(x,data[3])))
    if len(data) >= 6 :
        x4 = cc_mult(x2,x2)
        ret = add(ret,cc_mult(x4,cp_mult(x,data[5])))
        if len(data) >= 8 :
            x3 = cc_mult(x,x2)
            ret = add(ret,cc_mult(x4,cp_mult(x3,data[7])))
    return ret