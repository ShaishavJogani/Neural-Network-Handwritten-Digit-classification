import os
import numpy as np
from mnist import MNIST
import random

np.random.seed(7642)
mndata = MNIST('data')
images, labels = mndata.load_training()
index = random.randrange(0, len(images))

weights1 = 2*np.random.random((784, 256)) -1
bias1 = 2*np.random.random(256) - 1
weights2 = 2*np.random.random((256, 256)) -1
bias2 = 2*np.random.random(256) -1
weights3 = 2*np.random.random((256, 10)) -1
bias3 = 2 *np.random.random(10) -1
h1=None
h2=None
d1=None
d2=None
d3=None
dropout_percent = 0.8
dr1=None
dr2=None
trainSize = 10000
validSize = 5000
testSize = 5000
totalSize = trainSize + validSize + testSize
allrandom = np.random.choice(len(images), totalSize, False)

def oneHot(label):
    labelVector = [0]*10
    labelVector[label] = 1
    return labelVector

def getValidationData():
    imageList = []
    global allrandom,validSize
    randomNo = allrandom[0:validSize:1]
    for random in randomNo:
        temp = images[random]
        for i in range(len(temp)):
            temp[i] = float(temp[i])/255
        imageList.append( (temp, oneHot(labels[random])) )
    return imageList

def getTrainingData():
    imageList = []
    global allrandom, totalSize,validSize,testSize
    randomNo = allrandom[validSize+testSize:totalSize:1]
    for random in randomNo:
        temp = images[random]
        for i in range(len(temp)):
            temp[i] = float(temp[i])/255
        imageList.append( (temp, oneHot(labels[random])) )
    return imageList

def getTestData():
    imageList = []
    global allrandom,validSize,testSize
    randomNo = allrandom[validSize:validSize+testSize:1]
    for random in randomNo:
        temp = images[random]
        for i in range(len(temp)):
            temp[i] = float(temp[i])/255
        imageList.append( (temp, oneHot(labels[random])) )
    return imageList


def predictedOutput(input, doDropout = False):
    z1 = np.dot(weights1.T, input) + bias1
    global h1,h2,dr1,dr2
    h1 = 1/(1 + np.exp(-z1))    #sigmoid
    if(doDropout):
        #d1 = np.random.rand(h1.shape[0],h1.shape[1]) < dropout_percent
        #h1 = np.multiply(h1,d1)
        #h1/=dropout_percent
        dr1 = np.random.binomial(1, dropout_percent, size=h1.shape)
        #print h1
        h1 = np.multiply(h1, dr1)
        #print h1
    else:
        h1 *=dropout_percent
    # h1 *= np.random.binomial([np.ones((len(X),hidden_dim))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))
    z2 = np.dot(weights2.T, h1) + bias2
    h2 = 1/(1 + np.exp(-z2))   #sigmoid
    if(doDropout):
        dr2 = np.random.binomial(1, dropout_percent, size=h2.shape)
        #print d1
        #print h1
        h2 = np.multiply(h2, dr2)
    else: h2 *=dropout_percent

    z3 = np.dot(weights3.T, h2) + bias3
    output = np.exp(z3) / np.sum(np.exp(z3))
    # output *= np.random.binomial(1, dropout_percent, size=output.shape) / dropout_percent
    return output
    # print output

    # weight = weights1.T
    # lists = []
    # for row in weight[:1]:
    #     print len(row)
    #     sum = 0
    #     for i in range(len(row)):
    #         sum += row[i]*input[i]
    #     # print input
    #     h = np.dot(row,input)
    #     print h
    #     print sum
    #     lists.append(h)
    # # print (lists[0])

def getLoss(output, label):
    loss = -np.sum(label * np.log(output))
    return loss

def train (trainData):
    global i,weights1,weights2,weights3,eta,bias1,bias2,bias3,etadecay,dr1,dr2
    for data in trainData:
        i += 1
        label = data[1]
        image = data[0]
        output = predictedOutput(image, True)
        d1 = output - label
        dW3 = np.dot(h2.reshape((256,1)),d1.reshape((1,10)))
        dB3 = d1

        derv2 = h2 * (1-h2)

        d2 = np.dot(d1.reshape((1,10)),weights3.T) * derv2
        #d2 = np.multiply(d2,dr2)

        dW2 = np.dot(h1.reshape((256,1)),d2)
        dB2 = d2.reshape(256)


        derv1 = h1 * (1-h1)
        d3 = np.dot(d2, weights2.T) * derv1
        #d3 = np.multiply(d3,dr1)

        x = np.asarray(image)

        dW1 = np.dot(x.reshape((784,1)),d3)
        dB1 = d3.reshape(256)
        #print dB1

        if i%5000 ==0:print ('Training sample %s , eta: %s and Loss : %s' %(i,eta,getLoss(output, label)))
        #print bias2
        if i%5000 == 0:
            if eta <= 0.005: etadecay = 0.0001
            eta -= etadecay
            if eta <=0.0001: eta = 0.0001

        weights3 -= eta * dW3
        weights2 -= eta * dW2
        weights1 -= eta * dW1
        bias3 -= eta * dB3
        bias2 -= eta * dB2
        bias1 -= eta * dB1

        #print bias2
    total=0
    correct=0
    global validData
    for testdata in validData:
        out = predictedOutput(testdata[0])
        predictMax = np.argmax(out)
        realMax = np.argmax(testdata[1])
        if predictMax == realMax:
            correct += 1
        total += 1
    accuracy = (float(correct)/total) * 100
    print 'accuracy: ', accuracy

def save(filename='model1.npz'):
    np.savez_compressed(
        file=os.path.join(os.curdir, 'models', filename),
        weights1=weights1,
        weights2=weights2,
        weights3=weights3,
        bias1=bias1,
        bias2=bias2,
        bias3=bias3,
        eta = eta,
        etadecay = etadecay
    )

def load():
    global weights1, weights2, weights3, eta, bias1, bias2, bias3,etadecay
    npz_members = np.load(os.path.join(os.curdir, 'models', 'model1.npz'))
    weights1 = np.asarray(npz_members['weights1'])
    weights2 = np.asarray(npz_members['weights2'])
    weights3 = np.asarray(npz_members['weights3'])
    bias1 = np.asarray(npz_members['bias1'])
    bias2 = np.asarray(npz_members['bias2'])
    bias3 = np.asarray(npz_members['bias3'])
    eta = float(npz_members['eta'])
    etadecay = float(npz_members['etadecay'])

trainData = getTrainingData()
validData = getValidationData()
testSet = getTestData()
i =0
eta = 0.1
etadecay = 0.005


# Set it to true to train or false to test
if False:
    # Uncomment the following line to load from previously trained weights
    #load()
    total = 0
    correct = 0
    # accuracy before training
    for testdata in validData:
        out = predictedOutput(testdata[0])
        predictMax = np.argmax(out)
        realMax = np.argmax(testdata[1])
        if predictMax == realMax:
            correct += 1
        total += 1
    accuracy = (float(correct) / total) * 100
    print 'accuracy before training: ', accuracy

    for j in range(30):
        np.random.shuffle(trainData)
        train(trainData)
        save()

    # Saving the trained weights and bias to models folder. Need to create it
    save()
else:
    load()
    total = 0
    correct = 0
    conf = np.zeros((10,10), dtype=np.int)
    for testdata in testSet:
        out = predictedOutput(testdata[0])
        predictMax = np.argmax(out)
        realMax = np.argmax(testdata[1])
        if predictMax == realMax:
            correct += 1
        total += 1
        conf[realMax][predictMax] += 1
    accuracy = (float(correct) / total) * 100
    print 'accuracy of test: ', accuracy
    print 'The confusion matrix is as follows:'
    print conf
