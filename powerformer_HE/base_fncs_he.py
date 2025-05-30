import warnings, torch
import numpy as np
from liberate import fhe
from liberate.fhe import presets
from liberate.fhe.bootstrapping import ckks_bootstrapping as bs
warnings.filterwarnings(action='ignore')

keys = set()
rk = dict()

cnts = {'rot':0,'ccmult':0,'cpmult':0,'add':0,'ks':0,'rs':0}
tot_cnts = {'rot':0,'ccmult':0,'cpmult':0,'add':0,'ks':0,'rs':0}
s = 2**16//2
print('slot_size :',s)
engine,sk,pk,evk,gk,conjk,rotk_dict = None,None,None,None,None,None,None

def preset_bts() :
    global engine,sk,pk,evk,gk,conjk,rotk_dict
    params = presets.params["bootstrapping"]
    engine = fhe.ckks_engine(**params, verbose=True)

    sk = bs.create_secret_key_sparse(engine, h=192)
    pk = engine.create_public_key(sk)
    evk = engine.create_evk(sk)

    gk = engine.create_galois_key(sk)
    conjk = engine.create_conjugation_key(sk)
    rotk_dict = bs.create_bs_key(engine, sk)
    bs.create_cts_stc_const(engine)
    make_rot_keys()

def preset_golds() :
    global engine,sk,pk,evk,gk,conjk,rotk_dict
    params = presets.params["gold"]
    engine = fhe.ckks_engine(**params, verbose=True)

    sk = bs.create_secret_key_sparse(engine, h=192)
    pk = engine.create_public_key(sk)
    evk = engine.create_evk(sk)
    gk = engine.create_galois_key(sk)
    #make_rot_keys()

def set_level(x, lev) :
    return engine.level_up(x,lev)

def make_rot_keys() :
    global rk
    keys = [11264, 5632, 16896, 22528, 511, 4096, 28160, 28672, 24576, 512, 16384, -1, -2, 6643, -8, -16, -4, 13286, -32, 15360, 19929, 23296, 26572, -4081, -4082, 26752, -4083, -64, 30720, 13312, 4992, 12288, 11648, 18304, 24960, 31616, 1408, 7040, 12672, 23936, -8169, -8170, 27648, 18432, 26624, 2048, 9216, -12257, 29568, -12258, 21120, 16640, 256, 2816, 8448, 14080, 19712, 25344, 30976, 8192, -24522, 3328, 9984, 6656, -16345, -16346, 6144, 7, 29952, 21504, 1024, 6, 128, 20480, 1664, 8320, 14976, 21632, 28288, 4224, 9856, 15488, 32384, 19968, -20433, -20434, -20435, 3072, -24521, -33215, 64, -28609, -28610, -28611, 60, -28613, -28614, -28615, -28616, 55, -28612, -24523, -26572, -24525, 50, -24527, -24528, -24526, -24524, 45, -20436, -20437, -20438, -20439, 40, -19929, -20440, -16347, -16348, 35, -16350, -16351, -16352, -16349, 30, -12259, -12260, -12261, -13286, 25, -12264, -12263, -12262, -8171, 20, -8173, -8174, -8175, -8176, 15, -8172, -6643, -4084, -4085, 10, -4087, -4088, -4086, 1, 5, 4, 3, 2]
    for data in keys :
        rk[data] = engine.create_rotation_key(sk,-data)

def pre_pack(x,lev) :
    if len(x[0])==32768 :
        return torch.stack([plain_pack(x[i],lev) for i in range(len(x))])
    return torch.stack([pre_pack(x[i],lev) for i in range(len(x))])
        

def redecrypt(x,lev=0,mult=0) :
    x = engine.decrode(x, sk)
    if mult : x *= mult
    x = engine.encorypt(x, pk, lev)
    return x

def conjugate(x) :
    cnts['ks'] += 1
    return engine.conjugate(ct=x,conjk=conjk)

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
    return engine.clone(x)

def rot(x,l) :
    if l == 0 : return x
    cnts['rot'] += 1
    cnts['ks'] += 1
    return engine.rotate_single(x, rk[l])

def bootstrap(x) :
    return bs.bootstrap(engine=engine, ct=x, rot_key_dict=rotk_dict, evk=evk, conjk=conjk)

def p_rot(x,l) :
    x = x.flatten()
    return np.roll(x,-l)

def cc_mult(x, y, ks=True) :
    cnts['ccmult'] += 1
    if ks : cnts['ks'] += 1
    return engine.mult(x, y, evk, ks)

def cp_mult(x, y, rs=True, pad=True) :
    if rs : cnts['rs']+=1
    cnts['cpmult'] += 1
    return engine.mult(x, y, None, rs, pad)

def rescale(x) :
    cnts['rs'] += 1
    return engine.rescale(x)

def mult_i(x) :
    return engine.mult_i(x,False)

def ks(x) :
    cnts['ks'] += 1
    return engine.relinearize(x,evk)

def add(x,y) :
    cnts['add'] += 1
    return engine.add(x,y)

def sub(x,y):
    cnts['add'] += 1
    return engine.sub(x,y)

def row_wise_pack(x, lev=0) :
    x = x.flatten()
    return engine.encorypt(x, pk, level=lev)

def column_wise_pack(x, lev=0) :
    x2 = np.copy(x.T)
    for j in range(64) :
        for i in range(4) :
            x2[j*4+i,:] = x[:,i*64+j]
    x2 = x2.flatten()
    return engine.encorypt(x2, pk, lev)

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
    return engine.plain_pack(x,lev)[0].to('cpu')

def unpack(x,shape) :
    ret = engine.decrode(x, sk)
    return ret[:np.prod(shape)].reshape(shape)

def unpack2(x,shape) :
    ret = engine.decrode(x, sk)
    ret = ret[:np.prod(shape)].reshape(shape).T
    ret2 = np.copy(ret)
    for j in range(64) :
        for i in range(4) :
            ret2[:,i*64+j] = ret[:,j*4+i]
    return ret2
    
def unpack3(x,shape) :
    ret = engine.decrode(x, sk)
    ret = np.copy(ret[:np.prod(shape)].reshape(shape))
    x2 = np.copy(ret)
    return x2
