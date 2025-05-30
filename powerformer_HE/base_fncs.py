import warnings
import numpy as np
warnings.filterwarnings(action='ignore')

keys = set()
rk = dict()

cnts = {'rot':0,'ccmult':0,'cpmult':0,'add':0,'ks':0,'rs':0}
tot_cnts = {'rot':0,'ccmult':0,'cpmult':0,'add':0,'ks':0,'rs':0}
s = 2**16//2
print('slot_size :',s)
engine,sk,pk,evk,gk,conjk,rotk_dict = None,None,None,None,None,None,None

def set_level(x, lev) :
    return x

def pre_pack(x,lev) :
    return x

def redecrypt(x,lev=0,mult=0) :
    x = engine.decrode(x, sk)
    if mult : x *= mult
    x = engine.encorypt(x, pk, lev)
    return x

def conjugate(x) :
    cnts['ks'] += 1
    return np.conjugate(x)

def clear() :
    for key in cnts :
        cnts[key] = 0

def clear_key() :
    global keys
    keys = set()

def add_tot() :
    for key in cnts :
        tot_cnts[key] += cnts[key]

def clear_tot() :
    for key in tot_cnts :
        tot_cnts[key] = 0

def clone(x) :
    return np.copy(x)

def rot(x,l) :
    x = x.flatten()
    if l!=0 : 
        cnts['rot'] += 1
        cnts['ks'] += 1
    return np.roll(x,-l)

def bootstrap(x) :
    return x

def p_rot(x,l) :
    x = x.flatten()
    return np.roll(x,-l)

def cc_mult(x, y, ks=True) :
    cnts['ccmult'] += 1
    if ks : cnts['ks'] += 1
    return np.multiply(x,y)

def cp_mult(x, y, rs=True, pad=True) :
    if rs : cnts['rs']+=1
    cnts['cpmult'] += 1
    
    if pad and type(y)!=int and type(y)!=float and type(y)!=np.float32 and type(y)!=np.float64 and type(y)!=np.int64 and type(y)!=np.int64: 
        y = y[0]
    return np.multiply(x,y)

def rescale(x) :
    cnts['rs'] += 1
    return x

def mult_i(x) :
    return x * 1j

def ks(x) :
    cnts['ks'] += 1
    return x

def add(x,y) :
    cnts['add'] += 1
    return x + y

def sub(x,y):
    cnts['add'] += 1
    return x - y

def row_wise_pack(x, lev=0) :
    x = x.flatten()
    return x

def column_wise_pack(x, lev=0) :
    x2 = np.copy(x.T)
    for j in range(64) :
        for i in range(4) :
            x2[j*4+i,:] = x[:,i*64+j]
    x2 = x2.flatten()
    return x2

def bias_pack(x) :
    x2 = np.copy(x.T)
    for j in range(64) :
        for i in range(4) :
            x2[j*4+i,:] = x[:,i*64+j]
    x2 = x2.flatten()
    return x2

def p_row_wise_pack(x) :
    x = x.flatten()
    return x

def p_column_wise_pack(x) :
    x = x.T.flatten()
    return x

def plain_pack(x,lev) :
    return x

def unpack(x,shape) :
    ret = clone(x)
    return ret[:np.prod(shape)].reshape(shape)

def unpack2(x,shape) :
    ret = clone(x)
    ret = ret[:np.prod(shape)].reshape(shape).T
    ret2 = np.copy(ret)
    for j in range(64) :
        for i in range(4) :
            ret2[:,i*64+j] = ret[:,j*4+i]
    return ret2
    
def unpack3(x,shape) :
    ret = clone(x)#engine.decrode(x, sk)
    ret = np.copy(ret[:np.prod(shape)].reshape(shape))
    x2 = np.copy(ret)
    return x2
