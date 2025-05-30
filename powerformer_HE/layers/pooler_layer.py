from pooler_layers.layer1 import precompute_layer1,apply_layer1
from pooler_layers.tanh import apply_tanh
import joblib,os,time
import numpy as np
from base_fncs import *

class bert_pooler :
    def __init__(self,shape,verbose=True) :
        self.shape = shape
        self.verbose = verbose
        self.coef = joblib.load('./poly_eval/tanh_poly.pkl')

    def premake(self) :
        path = './pre_weights/pre_pooler.pkl'
        if os.path.exists(path) and not self.new:
            self.pre = joblib.load(path)
        else :
            self.pre = precompute_layer1(self.wp,self.bp)
            joblib.dump(self.pre,path)

    def make_weights(self,ws) :
        self.new = False
        path = './pre_weights/pre_pooler_weight.pkl'
        if os.path.exists(path) and ws is None : self.wp,self.bp = joblib.load(path)
        else :
            self.new = True
            self.wp = np.array(ws['bert.pooler.dense.weight'].cpu(),dtype=np.complex64).T
            self.bp = np.array(ws['bert.pooler.dense.bias'].cpu(),dtype=np.complex64).reshape(1,-1)
            joblib.dump((self.wp,self.bp),path)

    def tanh(self,x) :
        return np.tanh(x)
    
    def tanh_he(self,x) :
        x = [cp_mult(data,1/40) for data in x]
        for i in range(len(self.coef)) :
            x = [apply_tanh(data,self.coef[i]) for data in x]
            if i == 1 :
                if self.verbose : 
                    calc = np.concatenate([unpack3(data,self.shape) for data in x],axis=1)
                    print(f'bts4 input range : {np.max(np.abs(calc)),np.min(calc),np.max(calc)}')
                    print('bts4',len(x))
                l = 3
                for j in range(l) :
                    x[j] = sub(x[j],mult_i(x[j+l]))
                x = x[:l]
                st = time.time()
                x = [bootstrap(data) for data in x]
                ed = time.time()
                conj = [conjugate(data) for data in x]
                bef = [cp_mult(add(d1,d2),0.5) for d1,d2 in zip(x,conj)]
                aft = [cp_mult(mult_i(sub(d1,d2)),0.5) for d1,d2 in zip(x,conj)]
                x = bef+aft
        return x, ed-st

    def forward(self, x):
        st = time.time()
        #for gpu options
        #w,m = self.pre[0].to('cuda:0'),self.pre[2].to('cuda:0')
        #x = apply_layer1(x,(w,self.pre[1],m),self.verbose)

        #for cpu options
        x = apply_layer1(x,self.pre,self.verbose)
        ed = time.time()
        x,bts_time = self.tanh_he(x)
        #
        return x,[ed-st],[bts_time]

    def real(self,x) :
        first_token_tensor = x[0,:]
        pooled_output = first_token_tensor@self.wp + self.bp
        pooled_output = self.tanh(pooled_output)
        return pooled_output
