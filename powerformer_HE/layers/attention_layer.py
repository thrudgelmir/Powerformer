from attention_layers.layer1 import precompute_layer1,apply_layer1
from attention_layers.layer2 import precompute_layer2,apply_layer2
from attention_layers.layer3 import precompute_layer3,apply_layer3
from attention_layers.layer4 import precompute_layer4,apply_layer4
from attention_layers.softmax import precompute_softmax,apply_softmax

from base_fncs import *
import numpy as np
import os,joblib
import torch
import time

class attention :
    def __init__(self,shape,i,verbose=True) :
        self.num = i
        self.verbose = verbose
        self.shape = shape

    def premake(self) :
        path = f'./pre_weights/pre_attention_{self.num}.pkl'
        if os.path.exists(path) and not self.new :
            self.pre = joblib.load(path)
        else :
            self.pre = []
            self.pre.append(precompute_layer1([self.wq,self.wk,self.wv],self.bq,self.bk,self.bv))
            self.pre.append(precompute_layer2(self.shape))
            self.pre.append(precompute_layer3(self.shape,self.bts_range))
            self.pre.append(precompute_layer4([self.wo],self.bo))
            self.pre.append(precompute_softmax(self.rd))
            joblib.dump(self.pre,path)
        print(self.num,'attention precompute over')

    def make_weights(self, ws) :
        self.new = False
        path = f'./pre_weights/pre_attention_weight_{self.num}.pkl'
        if os.path.exists(path) and ws is None : self.wq,self.bq,self.wk,self.bk,self.wv,self.bv,self.wo,self.bo,self.rd = joblib.load(path)
        else :
            self.new = True
            self.wq = np.array(ws[f'bert.encoder.layer.{self.num}.attention.self.query.weight'].cpu(),dtype=np.complex64).T
            self.bq = np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.attention.self.query.bias'].cpu().view(1,-1),dtype=np.complex64),(128,1))
            self.wk = np.array(ws[f'bert.encoder.layer.{self.num}.attention.self.key.weight'].cpu(),dtype=np.complex64).T
            self.bk = np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.attention.self.key.bias'].cpu().view(1,-1),dtype=np.complex64),(128,1))
            self.wv = np.array(ws[f'bert.encoder.layer.{self.num}.attention.self.value.weight'].cpu(),dtype=np.complex64).T
            self.bv = np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.attention.self.value.bias'].cpu().view(1,-1),dtype=np.complex64),(128,1))
            self.wo = np.array(ws[f'bert.encoder.layer.{self.num}.attention.output.dense.weight'].cpu(),dtype=np.complex64).T
            self.bo = np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.attention.output.dense.bias'].cpu().view(1,-1),dtype=np.complex64),(128,1))
            self.rd = 1/(np.tile(np.array(ws[f'bert.encoder.layer.{self.num}.attention.self.batch_method.running_denominator'].cpu().view(12,128,1),dtype=np.complex64),(1,1,128))+1e-10)
            joblib.dump((self.wq,self.bq,self.wk,self.bk,self.wv,self.bv,self.wo,self.bo,self.rd),path)

    def set_bts_range(self, bts_range) :
        self.bts_range = bts_range

    def softmax(self,x) :
        x = (x+5)**5
        x = x*self.rd
        return x

    def forward(self,x,mask) :
        
        #for gpu options
        """
        w = [self.pre[0][0][0].to('cuda:0'),self.pre[0][0][1].to('cuda:0')]
        q,k,v = apply_layer1(x, [w] + self.pre[0][1:], self.verbose)

        pre = [data.to('cuda:0') for data in self.pre[1]]
        attention_scores = apply_layer2(q, k, self.shape, pre, self.verbose)

        attention_probs = apply_softmax(attention_scores,self.pre[4],mask)

        pre = [data.to('cuda:0') for data in self.pre[2]]
        context_layer,bts_time = apply_layer3(attention_probs, v, self.shape, pre, self.bts_range, self.verbose)

        w = self.pre[3][0][0].to('cuda:0')
        output = apply_layer4(context_layer, [[w]] + self.pre[3][1:], self.verbose)
        """
        #for cpu options
        st1 = time.time()
        q,k,v = apply_layer1(x, self.pre[0], self.verbose)
        ed1 = time.time()

        st2 = time.time()
        attention_scores = apply_layer2(q, k, self.shape, self.pre[1], self.verbose)
        ed2 = time.time()

        st3 = time.time()
        attention_probs = apply_softmax(attention_scores,self.pre[4],mask)
        ed3 = time.time()

        st4 = time.time()
        context_layer,bts_time = apply_layer3(attention_probs, v, self.shape, self.pre[2], self.bts_range, self.verbose)
        ed4 = time.time()

        st5 = time.time()
        output = apply_layer4(context_layer, self.pre[3], self.verbose)
        ed5 = time.time()
        #
        return output, [ed1-st1,ed2-st2,ed3-st3,ed4-st4,ed5-st5], bts_time
    
    def transpose_for_scores(self,x) :
        x = torch.tensor(x,dtype=torch.complex64)
        new_x_shape = x.size()[:-1] + (12, 64)
        x = x.view(new_x_shape)
        return x.permute(1, 0, 2)
    
    def real(self, x, mask) :
        q = x@self.wq + self.bq
        k = x@self.wk + self.bk
        v = x@self.wv + self.bv
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)     
        attention_scores = torch.matmul(q,k.transpose(-1, -2))/ 8
        attention_probs = self.softmax(attention_scores)
        attention_probs = attention_probs * mask
        context_layer = torch.matmul(attention_probs,v)
        bts = np.max(np.abs(np.array(context_layer)))
        context_layer = np.array(context_layer.permute(1, 0, 2).contiguous()).reshape(128,-1)
        final = context_layer@self.wo + self.bo
        return final,bts
