from __future__ import division
from __future__ import print_function

import time
import tensorflow.compat.v1 as tf
from gcn.utils import *
from gcn.models import GAutoencoder

def train(FLAGS,train_arr, test_arr,labels):
    train_acc=0
        # Settings
    
    # Load data
    adj, features, size_u, size_v, logits_train, logits_test, train_mask, test_mask = load_data(train_arr, test_arr,labels)

    # Some preprocessing
    # features = preprocess_features(features)
    if FLAGS.model == 'GAutoencoder':
        model_func = GAutoencoder
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    
    
    # Define placeholders
    tf.compat.v1.disable_eager_execution()
    placeholders = {
        'adjacency_matrix': tf.placeholder(tf.int32, shape=adj.shape),
        'Feature_matrix': tf.placeholder(tf.float32, shape=features.shape),
        'labels': tf.placeholder(tf.float32, shape=(None, logits_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'negative_mask': tf.placeholder(tf.int32)
    }
    
    # Create model
    model = model_func(placeholders, size_u, size_v, FLAGS.latent_factor_num)
    
    # Initialize session
    sess = tf.Session()
    
    
    # Define model evaluation function
    def evaluate(adj, features, labels, mask, negative_mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(adj, features,labels, mask, negative_mask, placeholders)
        outs = sess.run(model.outputs, feed_dict=feed_dict_val)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        outs=outs.reshape([533,89])
        return outs_val[0], outs_val[1], (time.time() - t_test),outs
    
    
    # Init variables
    sess.run(tf.global_variables_initializer())
    
    # Train model
    for epoch in range(FLAGS.epochs):
     
        t = time.time()
        # Construct feed dictionary
        negative_mask = generate_mask(labels, len(train_arr))
        
        feed_dict = construct_feed_dict(adj, features, logits_train, train_mask, negative_mask, placeholders)
             
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), 
              "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(1-outs[2]),
              "time=", "{:.5f}".format(time.time() - t))
#      
    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration,outs_test = evaluate(adj, features, logits_test, test_mask, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(1-test_acc), "time=", "{:.5f}".format(test_duration))