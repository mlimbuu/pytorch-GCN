import numpy as np
import time
import torch.nn.functional as F


"""Model Train and Test"""
# calulate accuracy from predicted labels and groundtruth
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# train the GCN model, default is single run with 200 epochs
def train_model(model, features, optimizer, epoch, adj, idx_train, idx_val, labels, type='single_train'):
    start = time.time()
    model.train()
    optimizer.zero_grad()
    out_features = model(features, adj)
    loss_train = F.nll_loss(out_features[idx_train], labels[idx_train])
    acc_train = accuracy(out_features[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    model.eval()
    out_features = model(features, adj)

    loss_val = F.nll_loss(out_features[idx_val], labels[idx_val])
    acc_val = accuracy(out_features[idx_val], labels[idx_val])
    print('Epoch: {:03d}'.format(epoch),
        'loss_train: {:.3f}'.format(loss_train.item()),
        'acc_train: {:.3f}'.format(acc_train.item()),
        'loss_val: {:.3f}'.format(loss_val.item()),
        'acc_val: {:.3f}'.format(acc_val.item()),
        'time: {:.3f}s'.format(time.time() - start))
    
    if type == 'single_train':
      return  acc_val, loss_val, out_features
    else:
      return acc_val, loss_val

 # test the GCN model     
def test_model(model, features, optimizer, adj, idx_test, labels):
    model.eval()
    out_features = model(features, adj)
    loss_test = F.nll_loss(out_features[idx_test], labels[idx_test])
    acc_test = accuracy(out_features[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.3f}".format(loss_test.item()),
          "accuracy= {:.3f}".format(acc_test.item()))
    return acc_test, loss_test


# runs the training for single run
"""###1. Without EarlyStopping and BatchNormalization"""
def single_run1(model, features, optimizer, adj, idx_train, idx_val, labels, epochs=200, earlyStopping=False):
  # Training the model
  start = time.time()
  val_acc_list = []
  val_loss_list = []
  for epoch in range(epochs):
      val_acc, loss_val, out_features = train_model(model, features, optimizer, epoch, adj, idx_train, idx_val, labels, type='single_train')
      val_acc_list.append(val_acc.item())
      val_loss_list.append(loss_val.item())
  print("Model Optimization Completed!")
  print("Total time elapsed: {:.3f}s".format(time.time() - start))

  # Test the model
  test_model()
  return val_acc_list, val_loss_list, out_features

"""####2. With EarlyStopping"""

def single_run2(model, features, optimizer, adj, idx_train, idx_val, labels, epochs=200, patience=10, earlyStopping=True):
  # Training the model
  start = time.time()
  val_acc_list = []
  val_loss_list = []
  best_val_acc = 0
  counter = 0

  for epoch in range(epochs):
      val_acc, loss_val, out_features = train_model(model, features, optimizer, epoch, adj, idx_train, idx_val, labels, type='single_train')
      val_acc_list.append(val_acc.item())
      val_loss_list.append(loss_val.item())

      # Check if validation accuracy has improved, and if not, increment the counter
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          counter = 0
      else:
          counter += 1

      # If the counter reaches the patience value, stop training and print message
      if counter == patience:
          print(f'Early stopping after {epoch+1} epochs with no improvement in validation accuracy.')
          break

  print("Model Optimization Completed!")
  print("Total time elapsed: {:.3f}s".format(time.time() - start))

  # Test the model
  test_model()
  return val_acc_list, val_loss_list, out_features

# single run output: val acc, val loos, outfeatures
val_acc_list, val_loss_list, out_features = single_run2()

"""####3. With batch normalization"""

# runs the training for single run
def single_run3(model, features, optimizer, adj, idx_train, idx_val, labels, epochs=200, earlyStopping=False):
  # Training the model
  start = time.time()
  val_acc_list = []
  val_loss_list = []
  for epoch in range(epochs):
      val_acc, loss_val, out_features = train_model(model, features, optimizer, epoch, adj, idx_train, idx_val, labels, type='single_train')
      val_acc_list.append(val_acc.item())
      val_loss_list.append(loss_val.item())
  print("Model Optimization Completed!")
  print("Total time elapsed: {:.3f}s".format(time.time() - start))

  # Test the model
  test_model()
  return val_acc_list, val_loss_list, out_features
 
 
"""### Multiple Runs Experiment
This experiment is only run when you need to calculate the average validation accuracy and validation loss for 100 runs.
"""

# runs the training and testing for 100 runs 
def multiple_runs(model, features, optimizer, epoch, adj, idx_train, idx_val, labels,iter=100, epochs=200):
    # validation avg outcome
    avg_val_acc_list = []
    avg_val_loss_list = []

    # test avg outcome
    avg_test_acc_list = []
    avg_test_loss_list = []

    for i in range(iter):
      # Training the model
      t_total = time.time()
      val_acc_list = []
      val_loss_list = []
      for epoch in range(epochs):
          val_acc, loss_val = train_model(model, features, optimizer, epoch, adj, idx_train, idx_val, labels, type='multiple_train')
          val_acc_list.append(val_acc.item())
          val_loss_list.append(loss_val.item())
      print("Model Optimization Completed!")
      print("Total time elapsed: {:.3f}s".format(time.time() - t_total))

      # Testing the model
      acc_test, loss_test = test_model()
      avg_test_acc_list.append(acc_test.item())
      avg_test_loss_list.append(loss_test.item())

    return avg_test_acc_list, avg_test_loss_list

avg_test_acc_list, avg_test_loss_list = multiple_runs()



  

