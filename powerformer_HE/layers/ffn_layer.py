from ffn_layers.layer1 import precompute_layer1,apply_layer1
from ffn_layers.layer2 import precompute_layer2,apply_layer2
from ffn_layers.gelu import apply_gelu
from base_fncs import *
import numpy as np
from scipy.special import erf
import torch,os,joblib,time

class ffn :
    def __init__(self,shape,i,verbose=True) :
        self.num = i
        self.verbose = verbose
        self.shape = shape
        self.coef = joblib.load('./poly_eval/gelu_poly.pkl')

    def premake(self) :
        path = f'./pre_weights/pre_ffn_{self.num}.pkl'
        if os.path.exists(path) and not self.new :
            self.pre = joblib.load(path)
        else :
            self.pre = []
            self.pre.append(precompute_layer1([self.wa],self.ba))
            self.pre.append(precompute_layer2([self.wb],self.bb))
            joblib.dump(self.pre,path)
        print(self.num,'ffn precompute over')

    def make_weights(self,ws) :
        self.new = False
        path = f'./pre_weights/pre_ffn_weight_{self.num}.pkl'
        if os.path.exists(path) and ws is None : 
            self.wa,self.ba,self.wb,self.bb = joblib.load(path)
        else :
            self.new = True
            self.wa = np.array(ws[f'bert.encoder.layer.{self.num}.intermediate.dense.weight'].cpu(),dtype=np.complex64).T
            self.ba = np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.intermediate.dense.bias'].cpu(),dtype=np.complex64),(128,1))
            self.wb = np.array(ws[f'bert.encoder.layer.{self.num}.output.dense.weight'].cpu(),dtype=np.complex64).T
            self.bb = np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.output.dense.bias'].cpu(),dtype=np.complex64),(128,1))
            joblib.dump((self.wa,self.ba,self.wb,self.bb),path)

    def gelu(self,x) :
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    
    def gelu_he(self,x) :
        x2 = [apply_gelu(cp_mult(data,2),self.coef[0]) for data in x]

        for i in range(1,4) :
            x2 = [apply_gelu(data,self.coef[i]) for data in x2]
            if i == 1 :
                l = 6
                for j in range(l) :
                    x2[j] = sub(x2[j],mult_i(x2[j+l]))
                x2 = x2[:l]
                st = time.time()
                x2 = [bootstrap(data) for data in x2]
                ed = time.time()
                conj = [conjugate(data) for data in x2]
                bef = [cp_mult(add(d1,d2),0.5) for d1,d2 in zip(x2,conj)]
                aft = [mult_i(cp_mult(sub(d1,d2),0.5)) for d1,d2 in zip(x2,conj)]
                x2 = bef+aft

        x = [cp_mult(data,40) for data in x]
        x = [cc_mult(d1,add(d2,1)) for d1,d2 in zip(x,x2)]
        return x, ed-st
    
    def forward(self, x) :

        st1 = time.time()
        #for gpu options
        #w = self.pre[0][0].to('cuda:0')
        #fc1 = apply_layer1(x, (w,self.pre[0][1],self.pre[0][2]), self.verbose)

        #for cpu options
        fc1 = apply_layer1(x, self.pre[0], self.verbose)
        ed1 = time.time() 
        #

        st2 = time.time()
        fc1, bts_time = self.gelu_he(fc1)
        ed2 = time.time() 

        
        #for gpu options
        #w = self.pre[1][0].to('cuda:0')
        #fc2 = apply_layer2(fc1, (w,self.pre[1][1],self.pre[1][2]), self.verbose)
        
        #for cpu options
        st3 = time.time()
        fc2 = apply_layer2(fc1, self.pre[1], self.verbose)
        ed3 = time.time()
        #
        return fc2, [ed1-st1, ed2-st2, ed3-st3], bts_time
    

    def real(self,x) :
        fc1 = x @ self.wa + self.ba
        fc1 = self.gelu(fc1)
        fc2 = fc1 @ self.wb + self.bb
        return fc2
