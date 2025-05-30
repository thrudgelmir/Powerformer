args = {
    'warmup' : 1000,
    'norm' : 10.0,
    'epoch' : 1000,
    'decay' : 0.01,
    'batch_size' : 64,
    'per_device_eval_batch_size' : 128,
    'lr' : 5e-5,
}


tasks = ['rte','mrpc','sst2','boolq']

baseline = {
    'pretrain':'google-bert/bert-base-uncased',
    'softmax':'softmax',
    'early_stopping':10,
    'load':True,
    'p':0,'c':0,'l':0,
    'distill':0,
    'step':0,
    'sl':128,
}

ours = {
    'pretrain':'google-bert/bert-base-uncased',
    'softmax':'power',
    'early_stopping':20,
    'load':True,
    'p':5,'c':5,'l':1.0,
    'distill':3,
    'step':0, 
    'sl':128,
}

#softmax :  "softmax" = original softmax,  "power" =  powered (x+c)^p
#distill :  1 = softmax only, 2 = ln only, 3 = both
#step :     0 = 1step loss O, 1 = 1step loss X, 2 = 2step