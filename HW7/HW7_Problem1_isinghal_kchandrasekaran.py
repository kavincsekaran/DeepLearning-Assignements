"""
   This code is based on tutorial https://www.tensorflow.org/versions/r1.3/get_started/mnist/beginners
"""

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import sys
import itertools
import numpy as np


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_bias(shape):
    return tf.Variable(tf.ones(shape))

def mnist_fully_connected_model(params, MODE):
    
    num_hidden_layers=1
    hidden_layer_size, learning_rate, minibatch_size, epoch = params
    
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
    y_ = tf.placeholder(tf.float32, [None, output_layer_size])
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    
    # Train
    for i in range(epoch):
        batch_xs, batch_ys = mnist.train.next_batch(minibatch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i > epoch - 30 or i % 1000 == 0:
            acc, loss=sess.run([accuracy, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
            print("Train Iteration: {}, Loss: {}, Accuracy: {}".format(i, loss, acc))

    if(MODE=="Tune"):
        return(sess.run([accuracy, cross_entropy], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))
    elif(MODE=="Test"):
        return(sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    
    


if __name__ == '__main__':
    if(len(sys.argv)>=2):
        MODE="Tune"
        MODE=sys.argv[1]
    else:
        print("No Mode specified. Running Test mode. Specify <script_name.py> Tune for finding best hyperparameters")
        MODE="Test"
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    if(MODE=="Tune"):
        hidden_layer_sizes=[200,300,500]
        learning_rates=[0.8, 0.5, 0.1]
        minibatch_sizes=[64,128]
        epochs=[15000, 20000]
        hyperparameters = [hidden_layer_sizes,learning_rates, minibatch_sizes, epochs]
        all_parameter_combinations=list(itertools.product(*hyperparameters))
        costs=[]
        accuracies=[]
        for parameter_combo in all_parameter_combinations:
            hidden_layer_size, learning_rate, minibatch_size, num_epoch = parameter_combo
            acc, cost=mnist_fully_connected_model(parameter_combo, MODE)
            costs.append(cost)
            accuracies.append(acc)
            print("Hidden Layer Size: {}, Learning Rate: {}, Minibatch Size: {}, Number of Epochs: {}, Validation Cost: {}, Validation Accuracy: {}".
              format(hidden_layer_size,learning_rate,minibatch_size,num_epoch,cost,acc))
                     
        best_params=all_parameter_combinations[np.argmin(costs)]
        print("Best Parameters: \n Hidden Layer Size: {}, Learning Rate: {}, Minibatch Size: {}, Number of Epochs: {}".
                format(best_params[0],best_params[1],best_params[2],best_params[3]))
        params=best_params
    else:
        params=(300,0.8,64,20000)
        
    acc, loss=mnist_fully_connected_model(params, MODE)
    print("Test Result: Loss: {}, Accuracy: {}".format(loss, acc))