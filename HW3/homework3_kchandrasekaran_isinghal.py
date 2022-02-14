# import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
import numpy as np

def J (w, faces, labels, alpha = 0.):
    y_hat=np.dot(np.transpose(w), np.transpose(faces))
    j_w=0
    for i in range(len(faces)):
        j_w+=(y_hat[i]-labels[i])**2
    return 0.5*j_w

def gradJ (w, faces, labels, alpha = 0.):
    y_hat=np.dot(np.transpose(w), np.transpose(faces))
    j_w=0
    for i in range(len(faces)):
        j_w+=((y_hat[i]-labels[i])**2)+(0.5*alpha*np.dot(w.T,w))
    return 0.5*j_w
 
    
def gradientDescent (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    w = np.random.randn(trainingFaces.shape[1])
    sigma=0.00001
    epsilon=1
    if(alpha>0):
        cost=gradJ(w, trainingFaces, trainingLabels, alpha)
    else:
        cost=J(w, trainingFaces, trainingLabels, alpha)
    while(True):
        grad=0
        y_hat=np.dot(np.transpose(w), np.transpose(trainingFaces))
        grad=np.dot((y_hat-trainingLabels), trainingFaces)
        new_w=w-epsilon*grad
        if(alpha>0):
            new_cost=gradJ(new_w, trainingFaces, trainingLabels, alpha)
        else:
            new_cost=J(new_w, trainingFaces, trainingLabels, alpha)
        print(abs(cost-new_cost))
        if(abs(cost-new_cost)<sigma):
            break
        w=new_w
        cost=new_cost
        
    return w

def sigma_z(z):
    return(1/(1+np.exp(-z)))

def J_CE(w, faces, labels, alpha=0.):
    y_hat=sigma_z(np.dot(faces,w))
    m=len(faces)
    j_ce=0
    #for i in range(len(faces)):
    #    print(labels[i]*np.log(y_hat[i]))
    #    print((1-labels[i])*np.log(1-y_hat[i]))
    #    print(0.5*alpha*np.dot(w.T,w))
        #print(labels[i]*np.log(y_hat[i])+(1-labels[i])*np.log(1-y_hat[i])+(0.5*alpha*np.dot(w.T,w)))
     #   j_ce+=(labels[i]*np.log(y_hat[i])+(1-labels[i])*np.log(1-y_hat[i]))
    #    print(j_ce)
    #j_ce=-(1/m)*j_ce
    #j_ce+=(0.5*alpha*np.dot(w.T,w))
    j_ce=np.sum(np.multiply(labels, np.log(y_hat+1e-9))+np.multiply((1-labels),np.log((1-y_hat)+1e-9)))+(0.5*alpha*np.dot(w.T,w))
    
    j_ce=-(j_ce/m)
    return j_ce

def method1 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    
    w=np.linalg.solve(np.dot(np.transpose(trainingFaces),trainingFaces),np.dot(np.transpose(trainingFaces),trainingLabels))
    return w

def method2 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels)

def method3 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 1e3
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha)

def method4(trainingFaces, trainingLabels):
    alpha = 1
    w = np.random.randn(trainingFaces.shape[1])
    cost=J_CE(w, trainingFaces, trainingLabels, alpha)
    sigma=0.000005
    epsilon=0.0001
    while(True):
        grad=0
        y_hat=sigma_z(np.dot(trainingFaces,w))
        grad=np.dot((y_hat-trainingLabels), trainingFaces)+alpha*w
        new_w=w-epsilon*grad
        new_cost=J_CE(new_w, trainingFaces, trainingLabels, alpha)
        #print(abs(cost-new_cost))
        print(new_cost)
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
    alpha=1e-2
    e_val, e_vec=np.linalg.eigh(np.dot(faces, faces.T)+alpha*np.identity(len(faces)))
    
    #print(np.diag(np.power(e_val, -0.5)))
    #whitened=np.dot(np.dot(np.power(np.diag(e_val), -0.5),e_vec.T), faces)
    #whitened=np.dot(np.dot(1/np.sqrt(np.diag(e_val)),e_vec.T), faces)
    #e_val=np.diag(e_val)
    #print(e_val)
    print(e_vec.shape)
    whitened=np.dot(np.dot(np.dot(e_vec,np.diag(np.power(e_val, -0.5))),e_vec.T), faces)
    #print(whitened.shape)
    #print(np.dot(whitened, whitened.T))
    w_e_val, w_e_vec=np.linalg.eigh(np.dot(whitened, whitened.T))
    #print(w_e_val)
    return whitened

if __name__ == "__main__":
    # Load data
    if ('trainingFaces' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        trainingFaces = np.load("trainingFaces.npy")
        trainingLabels = np.load("trainingLabels.npy")
        testingFaces = np.load("testingFaces.npy")
        testingLabels = np.load("testingLabels.npy")
    #trainingFaces=whiten(trainingFaces)
    #testingFaces=whiten(testingFaces)
    method4(trainingFaces, trainingLabels)
    
    #w1 = method1(trainingFaces, trainingLabels, testingFaces, testingLabels)
    #w2 = method2(trainingFaces, trainingLabels, testingFaces, testingLabels)
    #w3 = method3(trainingFaces, trainingLabels, testingFaces, testingLabels)
    #w4 = method3(trainingFaces, trainingLabels, testingFaces, testingLabels)
    #reportCosts(w4, trainingFaces, trainingLabels, testingFaces, testingLabels)
    #for w in [ w2, w3 ]:
    #    reportCosts(w, trainingFaces, trainingLabels, testingFaces, testingLabels)
    #detectSmiles(w3)  # Requires OpenCV
    