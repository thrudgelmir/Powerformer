from layers.attention_layer import attention
from layers.ffn_layer import ffn
from layers.add_norm_layer import add_norm
from layers.pooler_layer import bert_pooler
import joblib,os
from base_fncs import *
import numpy as np
import time

class bert_layer :
    def __init__(self,shape,i,verbose=True) :
        self.shape = shape
        self.verbose = verbose
        self.num = i
        self.attention = attention(shape,i,self.verbose)
        self.ffn = ffn(shape,i,self.verbose)
        self.add_norm1 = add_norm(i,0,self.verbose)
        self.add_norm2 = add_norm(i,1,self.verbose)

    def premake(self) :
        self.add_norm1.premake()
        self.add_norm2.premake()
        self.ffn.premake()
        self.attention.premake()

    def make_weights(self,ws=None) :
        self.add_norm1.make_weights(ws)
        self.add_norm2.make_weights(ws)
        self.ffn.make_weights(ws)
        self.attention.make_weights(ws)

    def set_bts_range(self, bts_range) :
        self.attention.set_bts_range(bts_range[0])
        self.add_norm2.set_bts_range(bts_range[1])

    def forward(self,x,mask) :
        #for gpu options
        #if x[0].level != 15 : 
        #    x = [set_level(data,15) for data in x]
        x2,t1,bts1 = self.attention.forward(x,mask)
        x,t2,bts2 = self.add_norm1.forward(x,x2)
        x2,t3,bts3 = self.ffn.forward(x)
        x,t4,bts4 = self.add_norm2.forward(x,x2)
        return x,np.concatenate([np.array(t1).flatten(),np.array(t2).flatten(),np.array(t3).flatten(),np.array(t4).flatten()]),[bts1,bts2,bts3,bts4]
    
    def real(self,x,mask) :
        x2,bts1 = self.attention.real(x,mask)
        x,_ = self.add_norm1.real(x,x2)
        x2 = self.ffn.real(x)
        x,bts2 = self.add_norm2.real(x,x2)
        return x,[bts1,bts2]


class bert_full :
    def __init__(self,shape,verbose=True) :
        self.shape = shape
        self.bert_layers = [bert_layer(shape,i,verbose) for i in range(12)]
        self.pooler = bert_pooler(shape,verbose)
        
    def premake(self) :
        for i in range(12) :
            self.bert_layers[i].premake()
        self.pooler.premake()

    def make_weights(self,ws=None) :
        for i in range(12) :
            self.bert_layers[i].make_weights(ws)
        self.pooler.make_weights(ws)

    def set_bts_range(self, inputs=None, masks=None) :
        path = './pre_weights/bts_range.pkl'
        if os.path.exists(path) and inputs is None :
            bts_range = joblib.load(path)
        else :
            bts_range = np.array([[0 for _ in range(2)]for _ in range(12)])
            for i in range(len(inputs)) :
                x = inputs[i,]
                mask = masks[i,]
                real,bts_now = self.real(x,mask)
                #print(np.max(np.abs(real.flatten() - output)))
                bts_range = np.maximum(bts_range,bts_now)
                if i % 50 == 0 : print(f'data {i} cmp')
            bts_range = np.array(bts_range).astype(np.int64)+1
            bts_range = np.where(bts_range%2==1,bts_range+1,bts_range)
            bts_range = [list(map(int,data)) for data in bts_range]
            joblib.dump(bts_range,path)
        
        for i in range(12) :
            self.bert_layers[i].set_bts_range(bts_range[i])

    def forward(self, x, mask) :
        layer_times = []
        bts_times = []
        for i in range(12) :
            x,layer_t,bts_time = self.bert_layers[i].forward(x,mask)
            layer_times.append(layer_t)
            bts_times.append(bts_time)
        layer_times = np.sum(np.array(layer_times),axis=0)
        bts_times = np.sum(np.array(bts_times),axis=0)
        x,pooler_t,pooler_bts = self.pooler.forward(x)
        return x, np.concatenate([layer_times,pooler_t]), np.concatenate([bts_times,pooler_bts])

    def real(self,x,mask) :
        bts_range = []
        for i in range(12) :
            x,bts = self.bert_layers[i].real(x,mask)
            bts_range.append(bts)
        x = self.pooler.real(x)
        return x,np.array(bts_range)

    def calc_diff(self, x, packed_x, mask) :
        real,bts_range = self.real(x,mask)
        st = time.time()
        calc, times, bts_time = self.forward(packed_x,mask)
        ed = time.time()
        calc = np.concatenate([unpack3(data,self.shape) for data in calc],axis=1)[:1,:]
        #print('real',real)
        #print('calc',calc)
        print('compute over')
        print(f'time cost : {ed-st}')
        print(f'max difference : {np.max(np.abs(real-calc))}')
