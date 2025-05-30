from models.approx_layers import GELU,LN
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.bert import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss
import torch, torch.nn as nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import math
import sys, os
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
sys.path.append(os.path.abspath('..'))

def search_layer(model,layer_types):
    ret = []
    for name, layer in model.named_children():
        for target_type in layer_types:
            if isinstance(layer, target_type):
                ret.append(layer)
        ret += search_layer(layer,layer_types)
    return ret

class BatchMethod(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer('running_denominator', torch.ones((1, 12, kwargs['sl'], 1), device=device))
        
    def forward(self, x):
        if self.training:
            batch_ret = torch.max(torch.sum(x,dim=-1,keepdims=True),dim=0,keepdims=True).values
            with torch.no_grad():
                self.running_denominator = batch_ret
        return self.running_denominator

class MyBertSelfAttention(nn.Module): 
    def __init__(self, config, **kwargs):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.a = 0

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        self.batch_method = BatchMethod(**kwargs)

        self.p, self.softmax = kwargs['p'], kwargs['softmax'].lower()
        self.c = kwargs['c']
        self.distill = kwargs['distill']

    def transpose_for_scores(self, x) :
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask) :
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (math.sqrt(self.attention_head_size)+1e-10)
        self.a = attention_scores
        if self.distill == 2 or self.softmax == 'softmax' :
            nor_mask = attention_mask.to(dtype=torch.float)
            nor_mask = (1.0 - nor_mask) * torch.finfo(nor_mask.dtype).min
            attention_max = torch.max(attention_scores,dim=-1,keepdim=True).values
            attention_probs = torch.exp(attention_scores+nor_mask-attention_max)
            attention_probs = attention_probs / (torch.sum(attention_probs,dim=-1,keepdim=True)+1e-10)

        else :
            attention_scores = attention_scores + self.c
            attention_probs = torch.pow(attention_scores,self.p)
            if self.distill == 1 or self.distill == 3 : attention_probs = attention_probs / (self.batch_method(attention_probs)+1e-10)
            else : attention_probs = attention_probs / (torch.sum(attention_probs,dim=-1,keepdim=True)+1e-10)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs * attention_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer

class MyBertSelfOutput(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LN(config.hidden_size, eps=config.layer_norm_eps, **kwargs)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor) :
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MyBertAttention(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.self = MyBertSelfAttention(config, **kwargs)
        self.output = MyBertSelfOutput(config, **kwargs)

    def forward(self, hidden_states, attention_mask) :
        attention_output = self.self(hidden_states, attention_mask)
        ln_output = self.output(attention_output, hidden_states)
        return ln_output


class MyBertOutput(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LN(config.hidden_size, eps=config.layer_norm_eps, **kwargs)
        self.h = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor) :
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        self.h = hidden_states
        return hidden_states

class MyBertIntermediate(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act_gelu = GELU(**kwargs)
        
    def forward(self, hidden_states: torch.Tensor) :
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_gelu(hidden_states)
        return hidden_states

class MyBertLayer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.attention = MyBertAttention(config, **kwargs)
        self.intermediate = MyBertIntermediate(config, **kwargs)
        self.output = MyBertOutput(config, **kwargs)

    def forward(self, hidden_states, attention_mask) :
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class MyBertPooler(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states) :
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MyBertEncoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([MyBertLayer(config, **kwargs) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask) :
        for layer in self.layer :
            layer_output = layer(hidden_states,attention_mask)
            hidden_states = layer_output
            
        return hidden_states


class MyBertModel(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = MyBertEncoder(config, **kwargs)
        self.pooler = MyBertPooler(config)
        self.e = None
        
    def binary_mask(self, attention_mask, input_shape):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        return extended_attention_mask
    
    def forward(self, input_ids, attention_mask, token_type_ids, save=False) :
        input_shape = input_ids.size()
        extended_attention_mask = self.binary_mask(attention_mask, input_shape)
        embedding_output = self.embeddings(input_ids, position_ids=None, token_type_ids=token_type_ids,inputs_embeds=None)
        self.e = embedding_output
        encoder_output = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(encoder_output)
        return pooled_output

class DistillLoss(nn.Module):
    def __init__(self, embedding_layers, teacher_embedding_layers, 
                 attention_layers, teacher_attention_layers,
                 hidden_layers, teacher_hidden_layers, 
                 ln_layers, teacher_ln_layers, config, distill, step):
        super().__init__()
        self.embedding_layers = [embedding_layers, teacher_embedding_layers]
        self.attention_layers = [attention_layers, teacher_attention_layers]
        self.hidden_layers = [hidden_layers, teacher_hidden_layers]
        self.ln_layers = [ln_layers, teacher_ln_layers]
        self.mse = MSELoss()
        self.cel = CrossEntropyLoss()
        self.config = config
        self.distill = distill
        self.step = step

    def soft_cross_entropy(self, predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()

    def forward(self, logits, teacher_logits, labels, problem_type, eval_pred) :
        if self.step < 2 :
            if problem_type == 'regression' :
                type_loss = self.mse(logits.squeeze(), labels.squeeze())
            else : 
                type_loss = self.soft_cross_entropy(logits, teacher_logits)
            embedding_loss = sum([self.mse(student.e,teacher.e) for student,teacher in zip(self.embedding_layers[0],self.embedding_layers[1])])
            attention_loss = sum([self.mse(student.a,teacher.a) for student,teacher in zip(self.attention_layers[0],self.attention_layers[1])])
            hidden_loss = sum([self.mse(student.h,teacher.h) for student,teacher in zip(self.hidden_layers[0],self.hidden_layers[1])])
            
            if self.distill >= 2 and self.step == 0: 
                ln_loss = sum([self.mse(student.l,teacher.l) for student,teacher in zip(self.ln_layers[0],self.ln_layers[1])])
            else : 
                ln_loss = 0
            loss = type_loss + embedding_loss + attention_loss/self.config.num_attention_heads + hidden_loss + ln_loss
        else :
            if eval_pred : 
                if problem_type == 'regression' :
                    loss = self.mse(logits.squeeze(), labels.squeeze())
                else : 
                    loss = self.soft_cross_entropy(logits, teacher_logits)
            else :
                embedding_loss = sum([self.mse(student.e,teacher.e) for student,teacher in zip(self.embedding_layers[0],self.embedding_layers[1])])
                attention_loss = sum([self.mse(student.a,teacher.a) for student,teacher in zip(self.attention_layers[0],self.attention_layers[1])])
                hidden_loss = sum([self.mse(student.h,teacher.h) for student,teacher in zip(self.hidden_layers[0],self.hidden_layers[1])])
                loss = embedding_loss + attention_loss + hidden_loss
        return loss
    
class CustomBert(nn.Module):
    def __init__(self, num_labels=2, **kwargs):
        orig = BertForSequenceClassification.from_pretrained(kwargs['pretrain'], num_labels=num_labels)
        config = orig.config
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.bert = MyBertModel(config, **kwargs)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        now = self.state_dict()
        new = {}
        for k, v in now.items() : 
            if k in orig.state_dict() : new[k] = orig.state_dict()[k]
            else : new[k] = now[k]
        self.load_state_dict(new)
        self.mse = MSELoss()
        self.cel = CrossEntropyLoss()
        
    def forward(self,input_ids,attention_mask,token_type_ids,labels) :
        pooled_output = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
                    
            if self.config.problem_type == "regression":
                loss = self.mse(logits.squeeze(), labels.squeeze())
            else:
                loss = self.cel(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
        

class CustomBertDistill(nn.Module):
    def __init__(self, num_labels, teacher, **kwargs):
        orig = BertForSequenceClassification.from_pretrained(kwargs['pretrain'], num_labels=num_labels) #original pretrained model
        config = orig.config
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.bert = MyBertModel(config, **kwargs)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        filtered = {k: (teacher.state_dict()[k] if k in teacher.state_dict() else v) for k, v in self.state_dict().items()}
        self.load_state_dict(filtered)

        self.loss_fct = DistillLoss(
            search_layer(self,[MyBertModel]),search_layer(teacher,[MyBertModel]),
            search_layer(self,[MyBertSelfAttention]),search_layer(teacher,[MyBertSelfAttention]),
            search_layer(self,[MyBertOutput]),search_layer(teacher,[MyBertOutput]),
            search_layer(self,[LN]),search_layer(teacher,[LN]),
            self.config, kwargs['distill'], kwargs['step'])

        self.teacher = teacher
        self.eval_pred = False
        
    def forward(self,input_ids,attention_mask,token_type_ids,labels) :
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids,attention_mask,token_type_ids,labels)
        pooled_output = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = self.loss_fct(logits,teacher_logits['logits'],labels,self.config.problem_type,self.eval_pred)
        return SequenceClassifierOutput(
            loss=loss,
            logits = logits
        )
        
