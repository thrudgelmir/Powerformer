import torch.nn as nn
import torch
import sys,os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.append(os.path.abspath('..'))

class BatchMethod(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer('running_denominator', torch.ones((1,kwargs['sl'],1), device=device))
        self.l = kwargs['l']
        
    def forward(self, x):
        if self.training:
            batch_ret = torch.max(torch.sqrt(x),dim=0,keepdims=True).values*self.l
            with torch.no_grad():
                self.running_denominator = batch_ret
        return self.running_denominator
    
    
class LN(torch.nn.Module): 
    def __init__(self, normalized_shape, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.batch_method = BatchMethod(**kwargs)
        self.distill = kwargs['distill']
        self.l = 0

    def forward(self, x):
        x = x - torch.mean(x, axis=-1, keepdims=True)
        variance = torch.var(x, axis=-1, keepdims=True)

        if self.distill >=2 : x_normalized = x / (self.batch_method(variance)+1e-10)
        else : x_normalized = x / (torch.sqrt(variance+1e-10)+1e-10)

        self.l = x_normalized

        x_normalized = self.weight * x_normalized + self.bias
        return x_normalized
    
        
class GELU(nn.Module) : 
    def __init__(self, **kwargs):
        super().__init__()
        self.gelu = nn.GELU()
        
    def forward(self, input) :
        return self.gelu(input)
    