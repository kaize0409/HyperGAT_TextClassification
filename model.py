import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from sklearn import metrics
from layers import *
from tqdm import tqdm


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer = False, concat=True)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer = True, concat=False)
        
    def forward(self, x, H):   
        x = self.gat1(x, H)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, H)

        return x


class DocumentGraph(Module):
    def __init__(self, opt, pre_trained_weight, class_weights, n_node, n_categories):
        super(DocumentGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.n_categories = n_categories
        self.batch_size = opt.batchSize
        self.dropout = opt.dropout    
        self.initial_feature = opt.initialFeatureSize
        self.dataset = opt.dataset


        self.embedding = nn.Embedding(self.n_node+1, self.initial_feature, padding_idx=0)
        self.layer_normH = nn.LayerNorm(self.hidden_size, eps=1e-6)

        self.prediction_transform = nn.Linear(self.hidden_size, self.n_categories, bias=True)  

        self.reset_parameters()

        if opt.dataset == 'mr':
            pre_trained_weight = torch.FloatTensor(pre_trained_weight)
            self.embedding = nn.Embedding.from_pretrained(pre_trained_weight, freeze = False, padding_idx = 0)

        self.hgnn = HGNN_ATT(self.initial_feature, self.initial_feature, self.hidden_size, dropout = self.dropout)

        self.class_weights = class_weights
        self.loss_function = nn.CrossEntropyLoss(weight = trans_to_cuda(torch.Tensor(self.class_weights).float()))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        
        

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        

    def compute_scores(self, inputs, node_masks):
        
        hidden = inputs * node_masks.view(node_masks.shape[0], -1, 1).float()
        b = torch.sum(hidden * node_masks.view(node_masks.shape[0], -1, 1).float(),-2)/torch.sum(node_masks,-1).repeat(hidden.shape[2],1).transpose(0,1)          
        b = self.layer_normH(b)           
        b = self.prediction_transform(b)

        pred = b
        
        return pred

    def forward(self, inputs, HT):
        
        hidden = self.embedding(inputs)
        nodes = self.hgnn(hidden, HT)
        return nodes


def forward(model, alias_inputs, HT, items, targets, node_masks):

    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    HT = trans_to_cuda(torch.Tensor(HT).float())
    node_masks = trans_to_cuda(torch.Tensor(node_masks).float())
    node = model(items, HT)
    get = lambda i: node[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, node_masks)


def train_model(model, train_data, opt):

    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(opt.batchSize, True)
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        alias_inputs, HT, items, targets, node_masks = train_data.get_slice(i)
        model.optimizer.zero_grad()
        targets, scores = forward(model, alias_inputs, HT, items, targets, node_masks)    
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets)
        loss.backward()
        model.optimizer.step()
        total_loss += loss

    print('\tLoss:\t%.4f' % (total_loss))



def test_model(model, test_data, opt, verbose=True):
    model.eval()

    test_pred = []
    test_labels = []
    slices = test_data.generate_batch(10, False)
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        alias_inputs, HT, items, targets, node_masks = test_data.get_slice(i)
        targets, scores = forward(model, alias_inputs, HT, items, targets, node_masks)
        pre_indices = scores.topk(1)[1]
        test_labels += list(targets)
        test_pred += list(trans_to_cpu(pre_indices).detach().numpy())

    details = metrics.classification_report(test_labels, test_pred, digits=4)
    acc = metrics.accuracy_score(test_labels, test_pred)

    if verbose:
        print("Test Precision, Recall and F1-Score...")
        print(metrics.classification_report(test_labels, test_pred, digits=4))
        print("Macro average Test Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
        print("Micro average Test Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

    return details,acc