from pylab import *
import random
from gcn.inits import *
import pandas as pd


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(train_arr, test_arr, labels):
    shape1 = 533
    shape2 = 89
    logits_test = sp.csr_matrix((labels[test_arr, 2], (labels[test_arr, 0] - 1, labels[test_arr, 1] - 1)),
                                shape=(shape1, shape2)).toarray()
    logits_test = logits_test.reshape([-1, 1])
    logits_train = sp.csr_matrix((labels[train_arr, 2], (labels[train_arr, 0] - 1, labels[train_arr, 1] - 1)),
                                 shape=(shape1, shape2)).toarray()
    logits_train = logits_train.reshape([-1, 1])
    train_mask = np.array(logits_train[:, 0], dtype=np.bool).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=np.bool).reshape([-1, 1])

    M = sp.csr_matrix((labels[train_arr, 2], (labels[train_arr, 0] - 1, labels[train_arr, 1] - 1)),
                      shape=(shape1, shape2)).toarray()
    DRS = pd.read_excel('../dataset/Disease_sim.xlsx', header=None).values
    CRS = pd.read_excel('../dataset/RNA_sim.xlsx', header=None).values
    adj = np.vstack((np.hstack((CRS, M)), np.hstack((M.transpose(), DRS))))
    CS = pd.read_csv("../dataset/re_CS.csv", header=None).values
    DS = pd.read_csv("../dataset/re_DS.csv", header=None).values
    features = np.vstack((np.hstack((CS, np.zeros(shape=(CS.shape[0], DS.shape[1]), dtype=int))),
                          np.hstack((np.zeros(shape=(DS.shape[0], CS.shape[1]), dtype=int), DS))))
    features = normalize_features(features)
    size_u = CS.shape
    size_v = DS.shape

    return adj, features, size_u, size_v, logits_train, logits_test, train_mask, test_mask

def generate_mask(labels, N):
    num = 0
    A = sp.csr_matrix((labels[:, 2], (labels[:, 0] - 1, labels[:, 1] - 1)), shape=(533, 89)).toarray()
    mask = np.zeros(A.shape)
    while (num < 1 * N):
        a = random.randint(0, 532)
        b = random.randint(0, 88)
        if A[a, b] != 1 and mask[a, b] != 1 and b != 10:
            mask[a, b] = 1
            num += 1
    mask = np.reshape(mask, [-1, 1])
    return mask


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized


def construct_feed_dict(adj, features, labels, labels_mask, negative_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['adjacency_matrix']: adj})
    feed_dict.update({placeholders['Feature_matrix']: features})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['negative_mask']: negative_mask})
    return feed_dict


def div_list(ls, n):
    ls_len = len(ls)
    j = ls_len // n
    ls_return = []
    for i in range(0, (n - 1) * j, j):
        ls_return.append(ls[i:i + j])
    ls_return.append(ls[(n - 1) * j:])
    return ls_return
