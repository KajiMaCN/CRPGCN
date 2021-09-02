import tensorflow.compat.v1 as tf
from gcn.train import *



# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'GAutoencoder', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.7, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 70e-6, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('latent_factor_num',65,'The size of latent factor vector.')

labels = np.loadtxt("../dataset/adj.csv")
reorder = np.arange(labels.shape[0])
np.random.shuffle(reorder)

T = 10
cv_num = 2
position_shape=47437

position2 = np.zeros([T, position_shape])
position3 = np.zeros([T, position_shape])
for t in range(T):
    order = div_list(reorder.tolist(),cv_num)
    for i in range(cv_num):
        test_arr = order[i]
        arr = list(set(reorder).difference(set(test_arr)))
        np.random.shuffle(arr)
        train_arr = arr
        train(FLAGS, reorder,test_arr,labels)