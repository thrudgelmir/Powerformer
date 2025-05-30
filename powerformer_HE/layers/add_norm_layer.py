from base_fncs import *
import numpy as np
import os,joblib
import time

class add_norm :
    def __init__(self,i,i2,verbose=True) :
        self.verbose = verbose
        self.num = i
        self.pos = i2
        self.bts_range = 1

    def premake(self) :
        path = f'./pre_weights/pre_ln_{self.pos}_{self.num}.pkl'
        if os.path.exists(path) and not self.new:
            self.pre,self.pre1 = joblib.load(path)
        else :
            rd = [bias_pack(self.rd[:,i*256:(i+1)*256]) for i in range(3)]
            weight = [bias_pack(self.weight[:,i*256:(i+1)*256]) for i in range(3)]
            self.pre = [rd[i]*weight[i]/768/self.bts_range for i in range(3)]
            self.pre1 = [bias_pack(self.bias[:,i*256:(i+1)*256])/self.bts_range for i in range(3)]
            if self.pos == 0 : 
                self.pre = pre_pack(self.pre,18)
            else :
                self.pre = pre_pack(self.pre,25)
            joblib.dump((self.pre,self.pre1),path)

    def make_weights(self, ws) :
        self.new = False
        path = f'./pre_weights/pre_ln_weight_{self.pos}_{self.num}.pkl'
        if os.path.exists(path) and ws is None : self.rd,self.weight,self.bias = joblib.load(path)
        else :
            self.new = True
            if self.pos == 0 :
                self.rd = 1/np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.attention.output.LayerNorm.rd.running_denominator'].cpu().view(128,1),dtype=np.complex64),(1,768))
                self.weight = np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.attention.output.LayerNorm.weight'].cpu(),dtype=np.complex64),(128,1))
                self.bias = np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.attention.output.LayerNorm.bias'].cpu().view(1,-1),dtype=np.complex64),(128,1))
            else : 
                self.rd = 1/np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.output.LayerNorm.rd.running_denominator'].cpu().view(128,1),dtype=np.complex64),(1,768))
                self.weight = np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.output.LayerNorm.weight'].cpu(),dtype=np.complex64),(128,1))
                self.bias = np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.output.LayerNorm.bias'].cpu().view(1,-1),dtype=np.complex64),(128,1))
            joblib.dump((self.rd,self.weight,self.bias),path)

    def set_bts_range(self, bts_range) :
        self.bts_range = bts_range

    def layernorm(self,x) :
        x -= np.mean(x,axis=-1,keepdims=True)
        x *= self.rd
        x = x*self.weight + self.bias
        return x
    
    def layernorm_he(self,x) :
        sum = add(add(x[0],x[1]),x[2])
        for i in range(8) :
            sum = add(sum,rot(sum,2**i*128))
        
        x = [cp_mult(x[i],768) for i in range(len(x))]
        x = [sub(x[i],sum) for i in range(len(x))]
        
        #for gpu options
        #w = [self.pre[i].to('cuda:0') for i in range(len(x))] 
        #x = [cp_mult(x[i],[w[i]]) for i in range(len(x))]
        
        #for cpu options
        x = [cp_mult(x[i],[self.pre[i]]) for i in range(len(x))]
        #

        x = [add(x[i],self.pre1[i]) for i in range(len(x))]
        return x
    
    def forward(self,befx,aftx) :
        st = time.time()
        tot = [add(bef,aft) for bef,aft in zip(befx,aftx)]
        tot = self.layernorm_he(tot)
        if self.pos == 1 : 
            tot[1] = sub(tot[1],mult_i(tot[2]))
            tot = tot[:2]
            bst = time.time()
            tot = [bootstrap(data) for data in tot]
            bed = time.time()
            conj = conjugate(tot[1])
            real = add(tot[1],conj)
            cplx = mult_i(sub(tot[1],conj))
            tot[1] = real
            tot.append(cplx)
            tot[0] = cp_mult(tot[0],self.bts_range)
            tot[1] = cp_mult(tot[1],self.bts_range//2)
            tot[2] = cp_mult(tot[2],self.bts_range//2)
            
        ed = time.time()
        bts_time = bed-bst if self.pos==1 else 0
        return tot, ed-st, bts_time
    
    def real(self,x1,x2) :
        x = self.layernorm(x1 + x2)
        bts = np.max(np.abs(x))
        return x,bts
