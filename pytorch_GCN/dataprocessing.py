import torch
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle as pkl

# normalize sparse adj matrix and features
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    row_inv = np.power(rowsum, -1).flatten()
    row_inv[np.isinf(row_inv)] = 0.
    row_mat_inv = sp.diags(row_inv)
    mx = row_mat_inv.dot(mx)
    return mx
# Convert a scipy sparse matrix to a torch sparse tensor.
def sparse_matrix_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# Parse index file
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
    
# load data for citseer, cora and pubmed with file extensions x,y,tx,ty,allx,ally,graph
def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # normalize features and adjmatrix
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # explicitly stated  train, val and test splits
  
    if dataset_str == 'citeseer':
        idx_train = range(120)
    elif dataset_str == 'cora':
        idx_train = range(140)
    elif dataset_str == 'pubmed':
        idx_train = range(60)
    else:
      idx_train = range(100)
    idx_val = range(200, 700)
    idx_test = range(700, 1700)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_matrix_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test

def random_split(num_total, num_train, num_val, num_test, shuffle=True):
    # Create a list of indices for the given range
    indices = list(range(num_total))
    # Shuffle the indices randomly if shuffle=True
    if shuffle:
        np.random.shuffle(indices)
    # Split the shuffled indices into train, validation, and test sets
    idx_train = indices[:num_train]
    idx_val = indices[num_train:num_train+num_val]
    idx_test = indices[num_train+num_val:num_train+num_val+num_test]
    return idx_train, idx_val, idx_test
