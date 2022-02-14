# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 18:28:24 2018

@author: Kavin Chandrasekaran, Ishaan Singhal
"""

# import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
import numpy as np

def J (w, faces, labels, alpha = 0.):
    y_hat=np.dot(faces,w)
    j_w=0
    j_w = (y_hat - labels)**2
    j_w = 0.5*np.sum(j_w)
    return j_w

def gradJ (w, faces, labels, alpha = 0.):
    y_hat=np.dot(faces,w)
    j_w=0
    j_w = (y_hat - labels)**2
    j_w = ((0.5*j_w) + (0.5*alpha*np.dot(w.T,w)))
    return j_w
 
def gradientDescent (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    w = np.random.randn(trainingFaces.shape[1])
    sigma=1e-2
    epsilon=1
    if(alpha>0):
        cost=gradJ(w, trainingFaces, trainingLabels, alpha)
    else:
        cost=J(w, trainingFaces, trainingLabels, alpha)
    i=0
    while(True):
        grad=0
        y_hat=np.dot(np.transpose(w), np.transpose(trainingFaces))
        grad = np.dot((y_hat-trainingLabels), trainingFaces) + (alpha*w)
        new_w = w - epsilon*grad
        if(alpha>0):
            new_cost=gradJ(new_w, trainingFaces, trainingLabels, alpha)
        else:
            new_cost=J(new_w, trainingFaces, trainingLabels, alpha)
        i+=1
        print "Tolerance Value (Target value", sigma, "): ", str(abs(cost-new_cost)), "iteration#", i
        print "Old Cost :", cost
        print "New Cost :", new_cost
        if(abs(cost-new_cost)<sigma):
            break
        w=new_w
        cost=new_cost
    return w

def sigma_z(z):
    return(1/(1+np.exp(-z)))

#Computing the value of the regularized cross-entropy loss function
def J_CE(w, faces, labels, alpha=0.):
    y_hat=sigma_z(np.dot(faces,w))
    m=len(faces)
    j_ce=0
    j_ce=np.sum(np.multiply(labels, np.log(y_hat+1e-9))+np.multiply((1-labels),np.log((1-y_hat)+1e-9)))
    j_ce=-(j_ce/m)+(0.5*alpha*np.dot(w.T,w))
    return j_ce

def method2 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels)

def method4(trainingFaces, trainingLabels):
    alpha=0
    w = np.random.randn(trainingFaces.shape[1])
    cost=J_CE(w, trainingFaces, trainingLabels, alpha)
    sigma=5e-7
    epsilon=2e-1
    m=len(trainingFaces)
    i=0
    while(True):
        grad=0
        y_hat=sigma_z(np.dot(trainingFaces,w))
        grad=(np.dot((y_hat-trainingLabels), trainingFaces)/m) + (alpha*w)
        new_w = w - epsilon*grad
        new_cost=J_CE(new_w, trainingFaces, trainingLabels, alpha)
        i+=1
        print "Tolerance Value (Target value", sigma, "): ", str(abs(cost-new_cost)), "iteration#", i
        print "Old Cost :", cost, "New Cost :", new_cost
        if(abs(cost-new_cost)<sigma):
            break
        w=new_w
        cost=new_cost
    return w

def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    print "Training cost: {}".format(J(w, trainingFaces, trainingLabels, alpha))
    print "Testing cost:  {}".format(J(w, testingFaces, testingLabels, alpha))

# Accesses the web camera, displays a window showing the face, and classifies smiles in real time
# Requires OpenCV.
def detectSmiles (w):
    # Given the image captured from the web camera, classify the smile
    def classifySmile (im, imGray, faceBox, w):
        # Extract face patch as vector
        face = imGray[faceBox[1]:faceBox[1]+faceBox[3], faceBox[0]:faceBox[0]+faceBox[2]]
        face = cv2.resize(face, (24, 24))
        face = (face - np.mean(face)) / np.std(face)  # Normalize
        face = np.reshape(face, face.shape[0]*face.shape[1])

        # Classify face patch
        yhat = w.dot(face)
        print yhat

        # Draw result as colored rectangle
        THICKNESS = 3
        green = 128 + (yhat - 0.5) * 255
        color = (0, green, 255 - green)
        pt1 = (faceBox[0], faceBox[1])
        pt2 = (faceBox[0]+faceBox[2], faceBox[1]+faceBox[3])
        cv2.rectangle(im, pt1, pt2, color, THICKNESS)

    # Starting video capture
    vc = cv2.VideoCapture()
    vc.open(0)
    faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")  # TODO update the path
    while vc.grab():
        (tf,im) = vc.read()
        im = cv2.resize(im, (im.shape[1]/2, im.shape[0]/2))  # Divide resolution by 2 for speed
        imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        k = cv2.waitKey(30)
        if k >= 0 and chr(k) == 'q':
            print "quitting"
            break

        # Detect faces
        faceBoxes = faceDetector.detectMultiScale(imGray)
        for faceBox in faceBoxes:
            classifySmile(im, imGray, faceBox, w)
        cv2.imshow("WebCam", im)

    cv2.destroyWindow("WebCam")
    vc.release()

def whiten(faces):
    alpha=1e-3
    e_val, e_vec=np.linalg.eigh(np.dot(faces.T, faces)+alpha*np.identity(np.shape(faces)[1]))
    whitened=np.dot(e_vec,np.diag(np.power(e_val, -0.5)))
    return whitened


if __name__ == "__main__":
    # Load data
    if ('trainingFaces' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        trainingFaces = np.load("trainingFaces.npy")
        trainingLabels = np.load("trainingLabels.npy")
        testingFaces = np.load("testingFaces.npy")
        testingLabels = np.load("testingLabels.npy")

    #Whitening
    L_x_train = whiten(trainingFaces)
    trainingFaces_w= np.dot(trainingFaces, L_x_train)
    testingFaces_w = np.dot(testingFaces, L_x_train)
    w1 = method2(trainingFaces_w, trainingLabels, testingFaces_w, testingLabels)
    reportCosts(w1, trainingFaces_w, trainingLabels, testingFaces_w, testingLabels)

    #Logistic Regression Method
    w4 = method4(trainingFaces, trainingLabels)
    #detectSmiles(w3)  # Requires OpenCV