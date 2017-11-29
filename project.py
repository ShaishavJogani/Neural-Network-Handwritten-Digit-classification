import os
import sys
import numpy as np
from mnist import MNIST
import random
import matplotlib.pyplot as plt


class NeuralNet:
    def __init__(self, seed, dropout_percent, training_size, validation_size, testing_size, epoch, training, validation, testing):
        np.random.seed(seed)
        mndata = MNIST('data')
        self.images, self.labels = mndata.load_training()
        self.index = random.randrange(0, len(self.images))

        self.doTraining = training
        self.doValidation = validation
        self.doTesting = testing

        self.weights1 = 2*np.random.random((784, 256)) -1
        self.bias1 = 2*np.random.random(256) - 1
        self.weights2 = 2*np.random.random((256, 256)) -1
        self.bias2 = 2*np.random.random(256) -1
        self.weights3 = 2*np.random.random((256, 10)) -1
        self.bias3 = 2 *np.random.random(10) -1
        self.h1=None
        self.h2=None
        self.d1=None
        self.d2=None
        self.d3=None

        self.dr1=None
        self.dr2=None
        self.trainSize = training_size
        self.validSize = validation_size
        self.testSize = testing_size
        self.totalSize = self.trainSize + self.validSize + self.testSize
        self.allrandom = np.random.choice(len(self.images), self.totalSize, False)

        self.dropout_percent = dropout_percent

        self.trainData = None
        self.validData = None
        self.testSet = None
        self.i = 0

        self.lossList = []
        self.epochList = []
        self.epochs = epoch
        self.lossSum = 0
        self.eta = 0.1
        self.etadecay = 0.005

    def oneHot(self, label):
        labelVector = [0]*10
        labelVector[label] = 1
        return labelVector

    def getValidationData(self):
        imageList = []

        randomNo = self.allrandom[0:self.validSize:1]
        for random in randomNo:
            temp = self.images[random]
            for i in range(len(temp)):
                temp[i] = float(temp[i])/255
            imageList.append( (temp, self.oneHot(self.labels[random])) )
        return imageList

    def getTrainingData(self):
        imageList = []
        # global allrandom, totalSize,validSize,testSize
        randomNo = self.allrandom[self.validSize + self.testSize : self.totalSize : 1]
        for random in randomNo:
            temp = self.images[random]
            for i in range(len(temp)):
                temp[i] = float(temp[i])/255
            imageList.append( (temp, self.oneHot(self.labels[random])) )
        return imageList

    def getTestData(self):
        imageList = []
        # global allrandom,validSize,testSize
        randomNo = self.allrandom[self.validSize : self.validSize + self.testSize : 1]
        for random in randomNo:
            temp = self.images[random]
            for i in range(len(temp)):
                temp[i] = float(temp[i])/255
            imageList.append( (temp, self.oneHot(self.labels[random])) )
        return imageList

    def predictedOutput(self, input, doDropout = False):
        z1 = np.dot(self.weights1.T, input) + self.bias1
        # global h1,h2,dr1,dr2
        self.h1 = 1/(1 + np.exp(-z1))    #sigmoid
        if(doDropout):
            self.dr1 = np.random.binomial(1, self.dropout_percent, size=self.h1.shape)
            self.h1 = np.multiply(self.h1, self.dr1)
        else:
            self.h1 *= self.dropout_percent

        z2 = np.dot(self.weights2.T, self.h1) + self.bias2
        self.h2 = 1/(1 + np.exp(-z2))   #sigmoid
        if(doDropout):
            self.dr2 = np.random.binomial(1, self.dropout_percent, size=self.h2.shape)
            self.h2 = np.multiply(self.h2, self.dr2)
        else:
            self.h2 *= self.dropout_percent

        z3 = np.dot(self.weights3.T, self.h2) + self.bias3
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

    def getLoss(self, output, label):
        loss = -np.sum(label * np.log(output))
        return loss

    def train (self, trainData, iteration):
        # global i,weights1,weights2,weights3,eta,bias1,bias2,bias3,etadecay,dr1,dr2,lossSum
        # i=0

        for data in trainData:
            self.i += 1
            label = data[1]
            image = data[0]
            output = self.predictedOutput(image, True)
            d1 = output - label
            dW3 = np.dot(self.h2.reshape((256,1)), d1.reshape((1,10)))
            dB3 = d1

            derv2 = self.h2 * (1 - self.h2)

            d2 = np.dot(d1.reshape((1,10)), self.weights3.T) * derv2

            dW2 = np.dot(self.h1.reshape((256,1)), d2)
            dB2 = d2.reshape(256)


            derv1 = self.h1 * (1 - self.h1)
            d3 = np.dot(d2, self.weights2.T) * derv1

            x = np.asarray(image)

            dW1 = np.dot(x.reshape((784,1)), d3)
            dB1 = d3.reshape(256)
            #print dB1
            # lossSum = lossSum + getLoss(output, label)
            # if i%1000 == 0:
            #     lossList.append(lossSum/1000)
            #     epochList.append( i )
            #     lossSum = 0
            if self.i%5000 ==0:print ('Training sample %s , eta: %s and Loss : %s' %(self.i, self.eta, self.getLoss(output, label)))
            #print bias2
            if self.i%5000 == 0:
                if self.eta <= 0.005: self.etadecay = 0.0001
                self.eta -= self.etadecay
                if self.eta <=0.0001: self.eta = 0.0001

            self.weights3 -= self.eta * dW3
            self.weights2 -= self.eta * dW2
            self.weights1 -= self.eta * dW1
            self.bias3 -= self.eta * dB3
            self.bias2 -= self.eta * dB2
            self.bias1 -= self.eta * dB1

            #print bias2
        total=0
        correct=0
        # global validData
        for testdata in self.validData:

            out = self.predictedOutput(testdata[0])
            predictMax = np.argmax(out)
            realMax = np.argmax(testdata[1])
            if predictMax == realMax:
                correct += 1
            total += 1

            self.lossSum = self.lossSum + self.getLoss(out, testdata[1])

        self.lossList.append(self.lossSum/5000)
        self.epochList.append( iteration+1 )
        self.lossSum = 0


        accuracy = (float(correct) / total) * 100
        print 'accuracy: ', accuracy

    def save(self, filename='model1.npz'):
        np.savez_compressed(
            file = os.path.join(os.curdir, 'models', filename),
            weights1 = self.weights1,
            weights2 = self.weights2,
            weights3 = self.weights3,
            bias1 = self.bias1,
            bias2 = self.bias2,
            bias3 = self.bias3,
            eta = self.eta,
            etadecay = self.etadecay
        )

    def load(self):
        # global weights1, weights2, weights3, eta, bias1, bias2, bias3,etadecay
        npz_members = np.load(os.path.join(os.curdir, 'models', 'model1.npz'))
        self.weights1 = np.asarray(npz_members['weights1'])
        self.weights2 = np.asarray(npz_members['weights2'])
        self.weights3 = np.asarray(npz_members['weights3'])
        self.bias1 = np.asarray(npz_members['bias1'])
        self.bias2 = np.asarray(npz_members['bias2'])
        self.bias3 = np.asarray(npz_members['bias3'])
        self.eta = float(npz_members['eta'])
        self.etadecay = float(npz_members['etadecay'])

    def loadData(self):
        self.trainData = self.getTrainingData()
        self.validData = self.getValidationData()
        self.testSet = self.getTestData()

    def run(self):
        # Set it to true to train or false to test
        if self.doTraining:
            # Uncomment the following line to load from previously trained weights
            self.load()
            total = 0
            correct = 0
            # accuracy before training
            for testdata in self.validData:
                out = self.predictedOutput(testdata[0])
                predictMax = np.argmax(out)
                realMax = np.argmax(testdata[1])
                if predictMax == realMax:
                    correct += 1
                total += 1
            accuracy = (float(correct) / total) * 100
            print 'accuracy before training: ', accuracy

            for j in range(self.epochs):
                np.random.shuffle(self.trainData)
                self.train(self.trainData, j)
                self.save()

            plt.plot(self.epochList, self.lossList, 'k', self.epochList, self.lossList, 'ro')
            plt.xlabel('Training samples')
            plt.ylabel('Loss')
            plt.show()
            # Saving the trained weights and bias to models folder. Need to create it
            self.save()
        else:
            self.load()
            total = 0
            correct = 0
            for testdata in self.testSet:
                out = self.predictedOutput(testdata[0])
                predictMax = np.argmax(out)
                realMax = np.argmax(testdata[1])
                if predictMax == realMax:
                    correct += 1
                total += 1
            accuracy = (float(correct) / total) * 100
            print 'accuracy of test: ', accuracy

def readCommand ( argv ):
    from optparse import OptionParser
    usageStr = """
        USAGE: python neuralNet.py <options>
        Eamples:    (1) python neuralNet.py
                        - Test the neural network
                    (2) python neuralNet.py --training
                    (3) python neuralNet.py --help
    """

    parser = OptionParser(usageStr)

    parser.add_option('-s', '--seed', dest='seed', type='int',
                        help="Random seed for numpy", default=1)
    parser.add_option('-d', '--dropout_percent', dest='dropout_percent', type='float',
                        help="Dropout Percent. Between 0.0 to 1.0", default=0.8)
    parser.add_option('-k', '--training_size', dest='training_size', type='int',
                        help="Training Data Size", default=10000)
    parser.add_option('-l', '--validation_size', dest='validation_size', type='int',
                        help="Validation Data Size", default=5000)
    parser.add_option('-m', '--testing_size', dest='testing_size', type='int',
                        help="Testing Data Size", default=5000)
    parser.add_option('-e', '--epoch', dest='epoch', type='int',
                        help="Total Number of iteration.", default=1)
    parser.add_option('--training', dest='training', action='store_true',
                        help="Test the accuracy on training samples.", default=False)
    parser.add_option('--validation', dest='validation', action='store_true',
                        help="Test the accuracy on validate samples.", default=False)
    parser.add_option('--testing', dest='testing', action='store_true',
                        help="Test the accuracy on testing samples.", default=True)

    (options, junkArgs) = parser.parse_args(argv)

    if len(junkArgs) != 0:
        raise Exception('Command line input is invalid: ' + str(junkArgs))
    if options.dropout_percent <= 0 or options.dropout_percent > 1:
        raise Exception('Drop Percentage must be between (0, 1].')
    if options.training_size <= 0 or options.training_size > 50000:
        raise Exception('Training data size must be between (0, 50000].')
    if options.validation_size <= 0 or options.validation_size > 10000:
        raise Exception('Validation data size must be between (0, 10000].')
    if options.testing_size <= 0 or options.testing_size > 10000:
        raise Exception('Testing data size must be between (0, 10000].')
    if (options.training_size + options.validation_size + options.testing_size) > 60000:
        raise Exception('Sum of training, validation, and testing data size must be <= 60000')
    #ToDo: check only one out of training, validate, testing

    args = dict()

    args['seed'] = options.seed
    args['dropout_percent'] = options.dropout_percent
    args['epoch'] = options.epoch
    args['training_size'] = options.training_size
    args['validation_size'] = options.validation_size
    args['testing_size'] = options.testing_size
    args['training'] = options.training
    args['validation'] = options.validation
    args['testing'] = options.testing

    return args

if __name__ == '__main__':
    args = readCommand( sys.argv[1:]) #Read Arguments
    neuralnetwork = NeuralNet(**args)
    neuralnetwork.loadData()
    neuralnetwork.run()


    pass
