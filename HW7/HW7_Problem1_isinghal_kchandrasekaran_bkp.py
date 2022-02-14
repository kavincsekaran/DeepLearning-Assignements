from __future__ import absolute_import
"""
   This code is based on tutorial https://www.tensorflow.org/versions/r1.3/get_started/mnist/beginners
"""

from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


def accuracy(text, sess, x, y, y_, test_data, test_labels):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(text, ':', sess.run(accuracy, feed_dict={x: test_data,  
                                        y_: test_labels}))

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_bias(shape):
    return tf.Variable(tf.ones(shape))

def mnist_fully_connected_model(params, MODE):
    params=(1,200,0.8,64,15000)
    
    num_hidden_layers, hidden_layer_size, learning_rate, minibatch_size, epoch = params
    
    input_layer_size=784
    output_layer_size=10

    x = tf.placeholder(tf.float32, [None, input_layer_size])
    
    W1 = init_weights([input_layer_size, hidden_layer_size])
    b1 = init_bias([hidden_layer_size])
    
    W2 = init_weights([hidden_layer_size, output_layer_size])
    b2 = init_bias([output_layer_size])
    
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    y = tf.nn.softmax(tf.matmul(h1, W2) + b2)


    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    sess = tf.InteractiveSession()# with tf.InteractiveSession() as sess:
    tf.global_variables_initializer().run()
    
    # Train
    for i in range(epoch):
        batch_xs, batch_ys = mnist.train.next_batch(minibatch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        text = str(i+1) + ' Train'
        if i > epoch - 21 or i % 200 == 0:
            accuracy(text, sess, x, y, y_, test_data=mnist.train.images, 
                                test_labels=mnist.train.labels)

    # Test trained model
    text = 'Final accuracy'
    acc=accuracy(text, sess, x, y, y_, test_data=mnist.test.images, test_labels=mnist.test.labels)
    return(cross_entropy, acc)
    # print('Reset TF graph...')
    # sess.close()
    # tf.reset_default_graph()
    
    


if __name__ == '__main__':

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # Import data
    '''
    num_hidden_layers=[1,2,3]
    hidden_layer_sizes=[200,300,500]
    learning_rates=[0.8, 0.5, 0.1]
    minibatch_sizes=[64,128]
    epochs=[15000, 20000]
    MODE="Tune"
    hyperparameters = [num_hidden_layers, hidden_layer_sizes,learning_rates, minibatch_sizes, epochs]
    all_parameter_combinations=list(itertools.product(*hyperparameters))
	costs=[]
	accuracies=[]
    for parameter_combo in all_parameter_combinations:
        cost, acc=main(parameter_combo, MODE)
        costs.append(cost)
        accuracies.append(acc)
    print(acc)
    '''
    params=()
    MODE="Test"
    mnist_fully_connected_model(params, MODE)