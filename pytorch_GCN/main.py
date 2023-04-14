
import torch
import torch.optim as optim
import numpy as np

from dataprocessing import load_data
from model import GCN
from train import single_run1, single_run2, single_run3, multiple_runs
from visualize import visualize_learnedFeature_tSNE, visualize_validation_performance

""" Parameters and Model Initialization: Any parameters change should be done here including
    selecting dataset from citseer, cora and pubmed.
"""

## hyperparameters initialization
seed = 42
hidden = 16
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
epochs = 200

# chocie of GPU using cuda or cpu
cuda = torch.cuda.is_available()
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
  torch.cuda.manual_seed(seed)

# load datasets
#dataset = 'citeseer'
dataset = 'cora'
#dataset = 'pubmed'
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset)

# model and optimizer
model = GCN(nfeatures=features.shape[1],
            nhidden_layers=hidden,
            nclass=labels.max().item() + 1,
            dropout= dropout)
optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

# add initialized model and data to cuda
if cuda:
  model.cuda()
  features = features.cuda()
  adj = adj.cuda()
  labels = labels.cuda()
  idx_train = idx_train.cuda()
  idx_val = idx_val.cuda()
  idx_test = idx_test.cuda()


# single run output: val acc, val loos, outfeatures
val_acc_list, val_loss_list, out_features = single_run1()

# single run output: val acc, val loos, outfeatures
val_acc_list, val_loss_list, out_features = single_run1()

# single run output: val acc, val loos, outfeatures
val_acc_list, val_loss_list, out_features = single_run2()

avg_test_acc_list, avg_test_loss_list = multiple_runs()
## This give avg val accuracy and loss
sum(avg_test_acc_list)/len(avg_test_acc_list), sum(avg_test_loss_list)/len(avg_test_acc_list)

"""####Visulaize Validation Performace"""

visualize_validation_performance(val_acc_list, val_loss_list)

"""####Visulaize Learned Represenatation"""

visualize_learnedFeature_tSNE(labels, out_features, dataset)