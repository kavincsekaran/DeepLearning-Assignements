import numpy as np
import time
import itertools

def softmax(z):
    return((np.exp(z)/np.sum(np.exp(z), axis=0)))

def relu(z):
    return(np.maximum(np.zeros(z.shape), z))

def relu_prime(z):
    return(np.greater(z, np.zeros(z.shape)).astype(int))
    
def J_CE(w1, b1, w2, b2, images, labels):
    z1=(np.dot(images, w1)+b1).T
    h1=relu(z1)
    z2=(np.dot(h1.T, w2)+b2).T
    y_hat=softmax(z2)
    m=len(images)
    j_ce=np.sum(np.multiply(labels.T, np.log(y_hat+1e-9)))
    j_ce=-(j_ce/m)
    accuracy = np.mean(np.equal(np.argmax(y_hat.T,axis=1),np.argmax(labels,axis=1)))
    return(j_ce, accuracy)

def stochastic_gradient_descent(w1, b1, w2, b2, images, labels, minibatch_size, epsilon):
    indices=np.arange(images.shape[0])
    np.random.shuffle(indices)
    for minibatch_indices in np.array_split(indices, images.shape[0]/minibatch_size):
        minibatch_x=images[minibatch_indices,:]
        minibatch_y=labels[minibatch_indices]
        
        z1=(np.dot(minibatch_x, w1)+b1).T
        h1=relu(z1)
        z2=(np.dot(h1.T, w2)+b2).T
        y_hat=softmax(z2)
        dj_dz2=y_hat.T-minibatch_y
        dj_dh1=np.dot(dj_dz2, w2.T)
        dh1_dz1=relu_prime(z1)
        
        '''
        print("z1",z1.shape)
        print("h1",h1.shape)
        print("z2",z2.shape)
        print("y_hat",y_hat.shape)
        print("dj_dz2",dj_dz2.shape)
        print("dj_dh1",dj_dh1.shape)
        print("dh1_dz1", dh1_dz1.shape)
        '''
        g=np.multiply(dj_dh1.T, dh1_dz1)
        dj_dw1=np.dot(g,minibatch_x).T
        dj_db1=np.sum(g,axis=1)
        dj_dw2=np.dot(dj_dz2.T,h1.T).T
        dj_db2=np.sum(dj_dz2,axis=0)
        '''
        print("g",g.shape)
        print("dj_dw1",dj_dw1.shape)
        print("dj_db1", dj_db1.shape)
        print("dj_dw2", dj_dw2.shape)
        print("dj_db2",dj_db2.shape)
        '''
        w1=w1-(epsilon*dj_dw1)
        b1=b1-(epsilon*dj_db1)
        w2=w2-(epsilon*dj_dw2)
        b2=b2-(epsilon*dj_db2)
        
    return(w1, b1, w2, b2)

    
def findBestHyperparameters(trainingImages, trainingLabels, validationImages,validationLabels):
    
    input_size=trainingImages.shape[1]
    output_size=trainingLabels.shape[1]

    hidden_layer_sizes=[50,100,200]
    learning_rates=[0.001]
    minibatch_sizes=[64,128]
    epochs=[10,15]
    hyperparameters = [hidden_layer_sizes,learning_rates,minibatch_sizes, epochs]
    all_parameter_combinations=list(itertools.product(*hyperparameters))
	costs=[]
	accuracies=[]
    for parameter_combo in all_parameter_combinations:
		hidden_layer_size, learning_rate, minibatch_size, num_epoch=parameter_combo
		w1 = 2 * (np.random.random(size = (input_size, hidden_layer_size)) / input_size**0.5) - 1./input_size**0.5
        b1 = 0.01 * np.ones(hidden_layer_size)
        w2 = 2 * (np.random.random(size = (hidden_layer_size, output_size)) / hidden_layer_size**0.5) - 1./hidden_layer_size**0.5
        b2 = 0.01 * np.ones(output_size)
        
        start_time=time.time()
        for i in range(num_epoch):
            new_w1, new_b1, new_w2, new_b2=stochastic_gradient_descent(w1, b1, w2, b2, trainingImages, trainingLabels, minibatch_size, learning_rate)
            w1, b1, w2, b2=new_w1, new_b1, new_w2, new_b2
            
        cost, acc=J_CE(w1, b1, w2, b2, validationImages, validationLabels)
        costs.append(cost)
        accuracies.append(acc)
        print("Hidden Layer Size: {}, Learning Rate: {}, Minibatch Size: {}, Number of Epochs: {}, Validation Cost: {}, Validation Accuracy: {}, Time Elapsed: {}".
              format(hidden_layer_size,learning_rate,minibatch_size,num_epoch,cost,acc,time.time()-start_time))
    
    best_params=all_parameter_combinations[np.argmin(costs)]
    print("Best Parameters: \n Hidden Layer Size: {}, Learning Rate: {}, Minibatch Size: {}, Number of Epochs: {}".
              format(best_params[0],best_params[1],best_params[2],best_params[3]))
    return best_params
 
    
if __name__ == "__main__":
    trainingImages = np.load("mnist_train_images.npy")
    trainingLabels = np.load("mnist_train_labels.npy")
    validationImages = np.load("mnist_validation_images.npy")
    validationLabels = np.load("mnist_validation_labels.npy")
    testingImages = np.load("mnist_test_images.npy")
    testingLabels = np.load("mnist_test_labels.npy")

    input_size=trainingImages.shape[1]
    output_size=trainingLabels.shape[1]

    #default hyperparameters

    hidden_layer_size=50
    num_epoch=10
    minibatch_size=64
    learning_rate=0.001
    
    # find best based on validation cost
    
    #hidden_layer_size, learning_rate, minibatch_size, num_epoch = findBestHyperparameters(trainingImages, trainingLabels, validationImages,validationLabels)
    
    #Train based on best parameters
    '''
    w1=np.random.randn(input_size, hidden_layer_size)
    b1=np.ones(hidden_layer_size)
    w2=np.random.randn(hidden_layer_size, output_size)
    b2=np.ones(output_size)
    '''
    w1 = 2 * (np.random.random(size = (input_size, hidden_layer_size)) / input_size**0.5) - 1./input_size**0.5
    b1 = 0.01 * np.ones(hidden_layer_size)
    w2 = 2 * (np.random.random(size = (hidden_layer_size, output_size)) / hidden_layer_size**0.5) - 1./hidden_layer_size**0.5
    b2 = 0.01 * np.ones(output_size)
    
    for i in range(num_epoch):
        new_w1, new_b1, new_w2, new_b2=stochastic_gradient_descent(w1, b1, w2, b2, trainingImages, trainingLabels, minibatch_size, learning_rate)
        cost, acc=J_CE(new_w1, new_b1, new_w2, new_b2, trainingImages, trainingLabels)
        print("Cost: {}, Accuracy: {}".format(cost, acc))
        w1, b1, w2, b2=new_w1, new_b1, new_w2, new_b2
        
    #test weights and biases on testing set after training
    
    test_cost, test_acc=J_CE(w1, b1, w2, b2, testingImages, testingLabels)
    print("Test Cost: {}, Test Accuracy: {}".format(test_cost, test_acc))
    