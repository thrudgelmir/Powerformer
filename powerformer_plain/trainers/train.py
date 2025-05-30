from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, BertTokenizer, TrainerCallback
from functools import partial
from datasets import load_dataset, load_metric
import torch,joblib,os,shutil
import gc
from models.bert_model import CustomBert, CustomBertDistill
from trainers.seed import seedval
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")    
import shutil
import numpy as np
from transformers import TrainingArguments, Trainer

class GcCallback(TrainerCallback):
    def __init__(self, output_dir, max_checkpoints=30):
        self.output_dir = output_dir
        self.max_checkpoints = max_checkpoints

    def on_epoch_end(self, args, state, control, **kwargs):
        gc.collect()
        checkpoints = [
            os.path.join(self.output_dir, d) for d in os.listdir(self.output_dir)
            if os.path.isdir(os.path.join(self.output_dir, d)) and d.startswith("checkpoint")
        ]
        
        checkpoints.sort(key=lambda x: os.path.getmtime(x))
        while len(checkpoints) > self.max_checkpoints:
            oldest_checkpoint = checkpoints.pop(0) 
            shutil.rmtree(oldest_checkpoint) 
            print(f"Deleted old checkpoint: {oldest_checkpoint}")

class EarlyStopping(TrainerCallback):
    def __init__(self, early_stopping_patience=1, first_metric='eval_accuracy',second_metric='eval_loss'):
        self.early_stopping_patience = early_stopping_patience
        self.metric_name = first_metric
        self.second_metric_name = second_metric
        self.best_metric = None
        self.second_metric = None
        self.patience_counter = 0
        self.metric_improved = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.best_metric = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_metric = metrics.get(self.metric_name)
        current_metric2 = metrics.get(self.second_metric_name)
        if current_metric is None or np.isnan(current_metric) or np.isinf(current_metric):
            return
        if self.best_metric is None:
            self.best_metric = current_metric
            self.second_metric = current_metric2
            self.patience_counter = 0
            control.should_save = True
            self.metric_improved = True
        else:
            if self._is_improvement(current_metric, current_metric2, self.best_metric, self.second_metric):
                self.best_metric = current_metric
                self.second_metric = current_metric2
                self.patience_counter = 0
                control.should_save = True
                self.metric_improved = True
            else:
                self.patience_counter += 1
                self.metric_improved = False
                if self.patience_counter >= self.early_stopping_patience:
                    control.should_training_stop = True

    def _is_improvement(self, current, current2, best, second):
        if current > best : return True
        if current == best :
            if current2 < second : return True
        return False
        
    def on_save(self, args, state, control, **kwargs):
        if self.metric_improved:
            self.last_checkpoint = os.path.join(
                args.output_dir, f'checkpoint-{state.global_step}'
            )
            state.best_model_checkpoint = self.last_checkpoint
            self.metric_improved = False


def model_for_task(task, **kwargs):
    model = CustomBert(num_labels=2, **kwargs)
    return model

def model_for_distill(task, teacher, **kwargs):
    model = CustomBertDistill(2, teacher, **kwargs)
    return model

def preprocess_function(examples, task):
    if task == 'mrpc':
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)
    if task == 'sst2':
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)
    if task == 'rte':
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)
    if task == 'boolq':
        return tokenizer(examples["passage"], examples["question"], truncation=True, padding="max_length", max_length=256)


def preprocess_dataset(dataset, tasks) :
    for task in tasks :
        dataset[task]['test'] = dataset[task]['validation']
    return dataset

def load_args(args, pos, metric) :
    training_args = TrainingArguments(  
        num_train_epochs = args['epoch'],   
        save_strategy = 'epoch',
        output_dir = pos,
        per_device_train_batch_size = args['batch_size'],
        per_device_eval_batch_size = args['per_device_eval_batch_size'],       
        warmup_steps = args['warmup'],
        weight_decay = args['decay'],
        load_best_model_at_end=True,
        logging_strategy='no',
        report_to=[],
        metric_for_best_model = metric,
        evaluation_strategy="epoch",
        fp16 = False,
        max_grad_norm = args['norm'],
        gradient_accumulation_steps = 1,
        learning_rate = args['lr'],
        disable_tqdm = True,
    )
    return training_args

def compute_metrics(eval_pred, task):
    if task == 'boolq' : metric = load_metric("super_glue", task)
    else : metric = load_metric("glue", task)
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def load_trainer(task, pos, model, train_args, datasets, early_stopping) :
    trainer = Trainer(
        model = model,
        args = train_args,
        train_dataset = datasets['train'],
        eval_dataset = datasets['test'],
        data_collator = data_collator,
        compute_metrics=partial(compute_metrics, task=task),
        callbacks=[EarlyStopping(early_stopping_patience=early_stopping,
                                 first_metric=train_args.metric_for_best_model,
                                 ),GcCallback(pos)],
    )
    return trainer


def make_dataset(tasks) :
    name = 'datasets.pkl'
    if os.path.exists(name) : 
        exist_tasks,datasets = joblib.load(name)
        if tasks == exist_tasks : return datasets
    datasets = {}
    for task in tasks : 
        if task == 'boolq' :
            datasets[task] = load_dataset('super_glue', task).map(partial(preprocess_function,task=task), batched=True)
        else :
            datasets[task] = load_dataset('glue', task).map(partial(preprocess_function,task=task), batched=True)
    datasets = preprocess_dataset(datasets, tasks)
    joblib.dump((tasks,datasets),name)
    return datasets

def make_path(task,args,**kwargs) :
    return f"./saves/{args['seed']}/{kwargs['pretrain']}/{task}/{kwargs['softmax']}/{kwargs['distill']}/{kwargs['step']}/{int((kwargs['l']*10-10))}/{kwargs['p']}/{kwargs['c']}/"

def search_layer(model,layer_types):
    ret = []
    for name, layer in model.named_children():
        for target_type in layer_types:
            if isinstance(layer, target_type):
                ret.append(layer)
        ret += search_layer(layer,layer_types)
    return ret

def param_copy(a,b) :
    b.data.copy_(a.data)
    if a.grad is not None:
        if b.grad is None:
            b.grad = torch.clone(a.grad).detach()
        else:
            b.grad.data.copy_(a.grad.data)
        
def evaluate(model,pos,task,train_args,dataset,early_stopping,save=True) :
    model.eval()
    test = dataset['test']
    trainer = load_trainer(task, pos, model, train_args, dataset, early_stopping)
    eval = trainer.evaluate(test)
    print('eval_cmp : ', eval)
    if save :
        os.makedirs(os.path.dirname(pos), exist_ok=True)
        print(f'saved {pos+"save.pt"}')
        torch.save(model.state_dict(),pos+'save.pt')
    return eval

def clear_chkpoint(train_args) :
    checkpoints = [f.path for f in os.scandir(train_args.output_dir) if f.is_dir() and "checkpoint" in f.name]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)

def train_model(model, dataset, task, args, **kwargs) :
    pos = make_path(task,args,**kwargs)
    train_args = load_args(args, pos, get_metric(task))
    
    if kwargs['step'] == 2 :
        if model.eval_pred == False : pos += '0/'
        else : pos += '1/'
        os.makedirs(pos, exist_ok=True)

    if kwargs['load'] and os.path.exists(pos+'save.pt'): 
        print(f'model loaded {pos+"save.pt"}')
        orig = torch.load(pos+'save.pt')
        now = model.state_dict()
        new = {}
        for k, v in now.items() : 
            if k in orig : new[k] = orig[k]
            else : new[k] = now[k]
        model.load_state_dict(new)
        eval = evaluate(model,pos,task,train_args,dataset,kwargs['early_stopping'],False)
    else :
        print('train state start')
        trainer = load_trainer(task, pos, model, train_args, dataset, kwargs['early_stopping'])
        trainer.train()
        eval = evaluate(model,pos,task,train_args,dataset,kwargs['early_stopping'])
        clear_chkpoint(train_args)

    return eval,model

def get_metric(task) :
    if task == 'cola' : return 'eval_matthews_correlation'
    if task == 'stsb' : return 'eval_pearson'
    return 'eval_accuracy'

def get_glue_res(dataset, task, args, **kwargs) :
    seedval(args['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args['warmup'] = len(dataset['train'])//args['batch_size']
    print('args', args)
    print('kwargs', kwargs)

    model = model_for_task(task,**kwargs).to(device)
    model.train()

    eval,model = train_model(model, dataset, task, args, **kwargs)
    return model, eval[get_metric(task)]


def get_distill_res(dataset, task, args, teacher, **kwargs) :
    seedval(args['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args['warmup'] = len(dataset['train'])//args['batch_size']
    
    print('args', args)
    print('kwargs', kwargs)

    model = model_for_distill(task, teacher,**kwargs).to(device)
    model.train()
    eval,model = train_model(model, dataset, task, args, **kwargs)
    if kwargs['step'] == 2 :
        model.eval_pred = True
        eval,model = train_model(model, dataset, task, args, **kwargs)
    return eval['eval_loss'] ,eval[get_metric(task)]