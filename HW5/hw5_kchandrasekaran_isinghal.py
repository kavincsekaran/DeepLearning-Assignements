import numpy as np

def softmax(z):
    return(np.exp(z).T/np.sum(np.exp(z), axis=1)).T

def relu(z):
    return(np.maximum(np.zeros(z.shape), z))

def calculate_h(w, b, x):
    z1=np.dot(faces, w)+b
    return(relu(z1))

def calculate_y_hat(w, b, h)
    z2=np.dot(h, w)+b
    return(softmax(z2))
    
def J_CE(w1, b1, w2, b2, faces, labels):
    y_hat=calculate_y_hat(w2, b2, calculate_h(w1, b1, faces))
    m=len(faces)
    j_ce=np.sum(np.multiply(labels, np.log(y_hat+1e-9)))
    j_ce=-(j_ce/m)
    return(j_ce)

def stochastic_gradient_descent(w1, b1, w2, b2, images, labels, minibatch_size):
    cost=J_CE(w, images, labels)
    epsilon=0.7
    for i in range(images.shape[0]/minibatch_size):
        minibatch_indices=np.random.randint(images.shape[0], size=minibatch_size)
        minibatch_x=images[minibatch_indices,:]
        minibatch_y=labels[minibatch_indices]
        while(True):
                grad=0
                grad=grad_JCE(w,images, labels)
                new_w = w - epsilon*grad
                new_cost=J_CE(new_w, images, labels)
                i+=1
                print("iteration#{}, Old Cost :{}, New Cost :{}, Accuracy :{}".format(i, cost, new_cost, accuracy(new_w, images, labels)))
                if(abs(cost-new_cost)<sigma):
                    break
                w=new_w
                cost=new_cost
    return w, b

    
def accuracy(w, faces, labels):
    y_hat=np.argmax(sigma_z(w, faces), axis=1)
    y=np.argmax(labels, axis=1)
    accuracy = np.sum(np.equal(y_hat,y))/(float(len(labels)))
    return accuracy

def findBestHyperparameters():
    return w1, w2, b1, b2
    
if __name__ == "__main__":
    trainingImages = np.load("mnist_train_images.npy")
    trainingLabels = np.load("mnist_train_labels.npy")
    validationImages = np.load("mnist_validation_images.npy")
    validationLabels = np.load("mnist_validation_labels.npy")
    testingImages = np.load("mnist_test_images.npy")
    testingLabels = np.load("mnist_test_labels.npy")

    #w1 = gradientDescent(trainingFaces, trainingLabels)
    z=np.mat([[1, 2],[-1, 0]])
    print(relu(z))