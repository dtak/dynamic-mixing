from __future__ import absolute_import
from __future__ import print_function
from autograd import grad
from math import sqrt

import autograd.numpy.random as npr
import autograd.numpy as np
import tensorflow as tf
import os


def neural_net(x, weights, biases):

    # Create mixture network
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


# def compare_predictions(ypred, targets):
#     count = 0
#     for k in range(ypred.shape[0]):
#         curr_pred = ypred[k,:].astype(int)
#         curr_target = targets[k,:].astype(int)
#         if ((curr_pred == curr_target)).all():
#             count += 1
#     print(count)

# def compare_obs_mean(ypred, targets):
#     #find all the indices where obs_mean >0
#     #find all the indices in ypred where obs_mean >0
#     #compare the indices to see how many match

#     target_inds = np.where(targets[:,0] < 0)[0]
#     ypred_inds = np.where(ypred[:,0] < 0)[0]
#     print(target_inds)
#     print(ypred_inds)
#     print("Total true negative", len(target_inds))
#     print("Total pred negative", len(ypred_inds))
#     print(np.where(ypred_inds == target_inds))

def predict(nn, inputs, x, init, model_path):

    saver = tf.train.Saver()
    # start the second session
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        ypred = sess.run(nn, {x:inputs})
        ypred = (ypred).astype(int)
    return ypred


def train_mixture_network(inputs, targets, num_input, num_output):

    learning_rate = 0.0001
    epochs = 5000
    # batch_size = 25
    model_path = "/tmp/model.ckpt"

    display_step = 10
    num_hidden1 = 40
    # num_hidden2 = 30
    print("Num input:", num_input)

    x = tf.placeholder("float32", [None, num_input], name='x') # create a tensor with any number of rows and obs dim inputs
    y = tf.placeholder("float32", [None, num_output], name='y')

    # Initialise weights
    weights = {
    'h1': tf.Variable(tf.random_normal([num_input, num_hidden1])),
    #'h2': tf.Variable(tf.random_normal([num_hidden1, num_hidden2])),
    'out': tf.Variable(tf.random_normal([num_hidden1, num_output]))
}

    # Initialise biases
    biases = {
    'b1': tf.Variable(tf.random_normal([num_hidden1])),
    #'b2': tf.Variable(tf.random_normal([num_hidden2])),
    'out': tf.Variable(tf.random_normal([num_output]))

}
    pred = neural_net(x, weights, biases)
    # Define loss and optimiser
    mse = tf.losses.mean_squared_error(pred, targets)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start a new TF session
    with tf.Session() as sess:

        # Run the initialiser
        sess.run(init)
        cost = 0
        for epoch in range(epochs):
            _ , c = sess.run([optimizer, mse], feed_dict={x:inputs, y:targets})
            cost += c
            #if epoch % 50 == 0:
                #print('Epoch: ', (epoch+1), 'cost=', '{:.3f}'.format(cost))
                #print(sqrt(sess.run(mse, feed_dict={x:inputs, y:obs_targets})))

        # Save model to disk
        save_path = saver.save(sess, model_path)

    ypred = predict(pred, inputs, x, init, save_path)
    return pred, x, init, model_path
