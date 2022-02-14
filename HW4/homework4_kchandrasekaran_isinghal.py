import numpy as np

def sigma_z(w, x):
    z=np.dot(x, w)
    return(np.exp(z).T/np.sum(np.exp(z), axis=1)).T

#Computing the value of the cross-entropy loss function
def J_CE(w, faces, labels):
    y_hat=sigma_z(w, faces)
    m=len(faces)
    #labels=np.reshape(labels, labels.size)
    #y_hat=np.reshape(y_hat, y_hat.size)
    j_ce=0
    j_ce=np.sum(np.multiply(labels, np.log(y_hat+1e-9)))
    j_ce=-(j_ce/m)
    return j_ce

def grad_JCE(w,faces, y):
    y_hat=sigma_z(w, faces)
    m=len(faces)
    grad=-(np.dot(faces.T, (y-y_hat))/m)
    return grad

def gradientDescent(trainingFaces, trainingLabels):
    #w = np.random.randn(trainingFaces.shape[1], trainingLabels.shape[1])
    w=np.zeros((trainingFaces.shape[1], trainingLabels.shape[1]))
    cost=J_CE(w, trainingFaces, trainingLabels)
    sigma=1e-4
    epsilon=0.7
    i=0
    while(True):
        grad=0
        grad=grad_JCE(w,trainingFaces, trainingLabels)
        new_w = w - epsilon*grad
        new_cost=J_CE(new_w, trainingFaces, trainingLabels)
        i+=1
        print("iteration#{}, Old Cost :{}, New Cost :{}, Accuracy :{}".format(i, cost, new_cost, accuracy(new_w, trainingFaces, trainingLabels)))
        if(abs(cost-new_cost)<sigma):
            break
        w=new_w
        cost=new_cost
    return w

def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels):
    print "Training cost: {}".format(J_CE(w, trainingFaces, trainingLabels))
    print "Testing cost:  {}".format(J_CE(w, testingFaces, testingLabels))
    print "Testing Accuracy: {}".format(accuracy(w, testingFaces, testingLabels))
    
def accuracy(w, faces, labels):
    y_hat=np.argmax(sigma_z(w, faces), axis=1)
    y=np.argmax(labels, axis=1)
    accuracy = np.sum(np.equal(y_hat,y))/(float(len(labels)))
    return accuracy

if __name__ == "__main__":
    # Load data
    if ('trainingFaces' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        trainingFaces = np.load("mnist_train_images.npy")
        trainingLabels = np.load("mnist_train_labels.npy")
        testingFaces = np.load("mnist_test_images.npy")
        testingLabels = np.load("mnist_test_labels.npy")

    w1 = gradientDescent(trainingFaces, trainingLabels)
    reportCosts(w1, trainingFaces, trainingLabels, testingFaces, testingLabels)
    
    