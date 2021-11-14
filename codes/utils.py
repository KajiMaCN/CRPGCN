from pylab import *
import random
from codes.inits import *


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


def load_data(train_arr, test_arr, labels, CS, DS, CF, DF):
    nc = CF.shape[0]
    nd = DF.shape[0]
    logits_test = sp.csr_matrix((labels[test_arr, 2], (labels[test_arr, 0] - 1, labels[test_arr, 1] - 1)),
                                shape=(nc, nd)).toarray()
    logits_test = logits_test.reshape([-1, 1])

    logits_train = sp.csr_matrix((labels[train_arr, 2], (labels[train_arr, 0] - 1, labels[train_arr, 1] - 1)),
                                 shape=(nc, nd)).toarray()
    logits_train = logits_train.reshape([-1, 1])

    train_mask = np.array(logits_train[:, 0], dtype=np.bool).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=np.bool).reshape([-1, 1])

    M = sp.csr_matrix((labels[train_arr, 2], (labels[train_arr, 0] - 1, labels[train_arr, 1] - 1)),
                      shape=(nc, nd)).toarray()
    adj = np.vstack((np.hstack((CS, M)), np.hstack((M.transpose(), DS))))

    features = np.vstack((np.hstack((CF, np.zeros(shape=(CF.shape[0], DF.shape[1]), dtype=int))),
                          np.hstack((np.zeros(shape=(DF.shape[0], CF.shape[1]), dtype=int), DF))))

    features = normalize_features(features)
    size_u = CF.shape
    size_v = DF.shape

    adj = preprocess_adj(adj)

    return adj, features, size_u, size_v, logits_train, logits_test, train_mask, test_mask, labels, nc, nd


def generate_mask(labels, N, nc, nd):
    num = 0
    A = sp.csr_matrix((labels[:, 2], (labels[:, 0] - 1, labels[:, 1] - 1)), shape=(nc, nd)).toarray()
    mask = np.zeros(A.shape)
    label_neg = np.zeros((5 * N, 2))
    while (num < 5 * N):
        a = random.randint(0, nc - 1)
        b = random.randint(0, nd - 1)
        if A[a, b] != 1 and mask[a, b] != 1:
            mask[a, b] = 1
            label_neg[num, 0] = a
            label_neg[num, 1] = b
            num += 1
    mask = np.reshape(mask, [-1, 1])
    return mask, label_neg


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


def ROC(outs, labels, test_arr, label_neg):
    scores = []
    for i in range(len(test_arr)):
        l = test_arr[i]
        scores.append(outs[int(labels[l, 0] - 1), int(labels[l, 1] - 1)])
    for i in range(label_neg.shape[0]):
        scores.append(outs[int(label_neg[i, 0]), int(label_neg[i, 1])])
    test_labels = np.ones((len(test_arr), 1))
    temp = np.zeros((label_neg.shape[0], 1))
    test_labels1 = np.vstack((test_labels, temp))
    test_labels1 = np.array(test_labels1, dtype=np.bool).reshape([-1, 1])
    return test_labels1, scores
