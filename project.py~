import numpy as np
from mnist import MNIST
import random

np.random.seed(1)
mndata = MNIST('Data')
images, labels = mndata.load_training()
index = random.randrange(0, len(images))
# print(MNIST.display(images[2]))
# print images[1]

weights1 = np.random.random((785, 256))
weights2 = np.random.random((256, 256))
weights3 = np.random.random((256, 10))
# print weights1

def oneHot(label):
    labelVector = [0]*10
    labelVector[label] = 1
    return labelVector

def getValidationData(value = 5000):
    imageList = []
    randomNo = np.random.choice(len(images), value, False)
    for random in randomNo:
        temp = images[random]
        temp.insert(0,1)
        imageList.append( (temp, oneHot(labels[random])) )
    return imageList

def getTrainingData(value = 10000):
    imageList = []
    randomNo = np.random.choice(len(images), value, False)
    for random in randomNo:
        temp = images[random]
        temp.insert(0,1)
        imageList.append( (temp, oneHot(labels[random])) )
    return imageList

def getTestData(value = 5000):
    imageList = []
    randomNo = np.random.choice(len(images), value, False)
    for random in randomNo:
        temp = images[random]
        temp.insert(0,1)
        imageList.append( (temp, oneHot(labels[random])) )
    return imageList

# print (getValidationData(1)[0][1])

def predictedOutput(input):
    h1 = np.dot(weights1.T, input)
    l1 = 1/(1 + np.exp(-h1))    #sigmoid

    h2 = np.dot(weights2.T, l1)
    l2 = 1/(1 + np.exp(-h2))   #sigmoid

    o3 = np.dot(weights3.T, l2)
    output = np.exp(o3) / np.sum(np.exp(o3))
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

trainData = getTrainingData(1)
for data in trainData:
    label = data[1]
    image = data[0]
    output = predictedOutput(image)
    getLoss(output, label)
