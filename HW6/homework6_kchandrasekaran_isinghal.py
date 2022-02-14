import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize  # For check_grad, approx_fprime
import operator

class RNN:
    def __init__ (self, numHidden, numInput, numOutput, numTimesteps):
        self.numHidden = numHidden
        self.numInput = numInput
        self.lookback = numTimesteps
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.W = np.random.randn(numHidden) * 1e-1
        
    def backward (self, x, y):
        dJ_dU=np.zeros(self.U.shape)
        dJ_dV=np.zeros(self.V.shape)
        dJ_dW=np.zeros(self.W.shape)
        y_hat, h=rnn.forward(x)
        dJ_dy_hat=y_hat-y
        dy_hat_dh=self.W
        dh_dz=1-h**2
        #print(dh_dz.shape)
        #print(dy_hat_dh.shape)
        dJ_dW=np.dot(dJ_dy_hat.T, h)
        #g=np.outer(dJ_dy_hat, dy_hat_dh)
        g=np.multiply(np.outer(dJ_dy_hat, dy_hat_dh), dh_dz)
        for t in range(len(y)):
            delta=g[t]
            for lb in np.arange(max(0, t-self.lookback), t+1)[::-1]:
                dJ_dU += np.outer(delta, h[lb-1])
                dJ_dV[:,x[lb]-1] += delta
                delta=self.U.T.dot(delta)*(1-h[lb-1]**2)
        return(dJ_dU, dJ_dV, dJ_dW)
    
    def forward (self, x):
        y_hat=[]
        h=np.zeros((len(x), self.numHidden))
        h[-1]=np.zeros(self.numHidden)
        for t, x_t in enumerate(x):
            z=np.dot(self.U, h[t-1])+np.dot(self.V.T, x_t)
            h[t]=np.tanh(z)
        y_hat=np.dot(h, self.W)
        return(y_hat, h)
    
    def loss(self, x, y):
        y_hat, _=rnn.forward(x)
        return(np.sum((y_hat-y)**2)*0.5)

    def gradient_check(self, x, y, h = 0.1, error_threshold = 1e-3):
        # calculate the gradient using backpropagation
        bptt_gradients = self.backward(x, y)
        # list of all params we want to check
        model_parameters = ["U", "V", "W"]
        # gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # get the actual parameter value from model, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print("performing gradient check for parameter %s with size %d. " %(pname, np.prod(parameter.shape)))
            # iterate over each element of the parameter matrix, e.g. (0,0), (0,1)...
            it = np.nditer(parameter, flags = ['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # save the original value so we can reset it later
                original_value = parameter[ix]
                # estimate the gradient using (f(x+h) - f(x-h))/2h
                parameter[ix] = original_value + h
                gradplus = self.loss(x, y)
                parameter[ix] = original_value - h
                gradminus = self.loss(x, y)
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # reset parameter to the original value
                parameter[ix] = original_value
                # the gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate the relative error (|x - y|)/(|x|+|y|)
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # if the error is too large fail the gradient check
                if relative_error < error_threshold:
                    print("Gradient check error: parameter = %s ix = %s" %(pname, ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative error: %f" % relative_error)
                    return
                it.iternext()
            print("Gradient check for parameter %s passed. " %(pname))


# From https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
def generateData ():
    total_series_length = 50
    echo_step = 2  # 2-back task
    batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)

if __name__ == "__main__":
    xs, ys = generateData()
    numHidden = 6
    numInput = 1
    numTimesteps = 3
    epoch=1000
    rnn = RNN(numHidden, numInput, 1, numTimesteps)
    learning_rate=1e-2
    #rnn.gradient_check([1,0,1,0], [0,0,1,0])
    #scipy.optimize.check_grad(rnn.loss,rnn.backward, xs, ys )
    #'''
    for i in np.arange(0,epoch):
        dJ_dU, dJ_dV, dJ_dW=rnn.backward(xs, ys)
        rnn.U-=learning_rate*dJ_dU
        rnn.V-=learning_rate*dJ_dV
        rnn.W-=learning_rate*dJ_dW
    loss=rnn.loss(xs, ys)
    print("Cost :",loss)
    y_hat, _=rnn.forward(xs)
    print(np.around(abs(y_hat)))
    print(ys)
    #'''