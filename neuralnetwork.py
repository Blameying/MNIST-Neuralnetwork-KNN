from mnistLoader import *
import random
import numpy as np
from PyQt5.QtWidgets import QLabel

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

class NNetWork(object):
    
    def __init__(self,inputLayerNum,hideLayerNum,outputLayerNum,learnSpeed):
        self.hideLayer = [np.random.randn() for i in range(hideLayerNum)]
        self.hideLayer_bias = np.zeros(hideLayerNum,dtype=np.float32)
        self.outputLayer = [np.random.randn() for i in range(outputLayerNum)]
        self.outputLayer_bias = np.zeros(outputLayerNum,dtype=np.float32)
        self.ihWeights = np.random.randn(inputLayerNum,hideLayerNum)
        self.ihWeights_change = np.zeros((inputLayerNum,hideLayerNum),dtype=np.float32)
        self.hoWeights = np.random.randn(hideLayerNum,outputLayerNum)
        self.hoWeights_change = np.zeros((hideLayerNum,outputLayerNum),dtype=np.float32)
        self.learnSpeed = learnSpeed

    def train(self,src,epoche,mini_batch_size,qlabel):
        train_length = len(src)
        for now in range(epoche):
            qlabel.setText("正在训练: epoche: "+str(now))
            random.shuffle(src)
            mini_batch = [src[k:k+mini_batch_size] for k in range(0,train_length,mini_batch_size)]
            self.mini_batch_calculator(mini_batch,mini_batch_size)

    def mini_batch_calculator(self,mini_batch,mini_batch_size):
        for batch in mini_batch:
            for data in batch:
                mulResultI2H = data[0].reshape(1,len(data[0]))@self.ihWeights
                hideLayerOutput = sigmoid(mulResultI2H-self.hideLayer)
                mulResultH2O = hideLayerOutput.reshape(1,len(self.hideLayer))@self.hoWeights 
                outputLayerOutput = sigmoid(mulResultH2O-self.outputLayer)
                error = data[1]-outputLayerOutput

                middle_temp = outputLayerOutput*(1-outputLayerOutput)*error

                temp = self.learnSpeed*middle_temp
                self.hoWeights_change+=hideLayerOutput.reshape(len(self.hideLayer),1)@temp
                self.outputLayer_bias+=(-temp).reshape(len(self.outputLayer),)

                temp = (self.learnSpeed*hideLayerOutput*(1-hideLayerOutput)).reshape(len(self.hideLayer),1)*(self.hoWeights@middle_temp.reshape(len(self.outputLayer),1))
                self.ihWeights_change+=data[0].reshape(len(data[0]),1)@temp.reshape(1,len(self.hideLayer))
                self.hideLayer_bias+=(-temp).reshape(len(self.hideLayer),)
            
            self.hoWeights += self.hoWeights_change / mini_batch_size 
            self.outputLayer += self.outputLayer_bias / mini_batch_size 
            self.ihWeights += self.ihWeights_change / mini_batch_size
            self.hideLayer += self.hideLayer_bias / mini_batch_size 
            inputLayerNum = len(data[0])
            hideLayerNum = len(self.hideLayer)
            outputLayerNum = len(self.outputLayer)
            self.hoWeights_change = np.zeros((hideLayerNum,outputLayerNum),dtype=np.float32)
            self.ihWeights_change = np.zeros((inputLayerNum,hideLayerNum),dtype=np.float32)
            self.hideLayer_bias = np.zeros(hideLayerNum,dtype=np.float32)
            self.outputLayer_bias = np.zeros(outputLayerNum,dtype=np.float32)
    def test(self,src):
        mulResultI2H = src.reshape(1,len(src))@self.ihWeights
        hideLayerOutput = sigmoid(mulResultI2H-self.hideLayer)
        mulResultH2O = hideLayerOutput.reshape(1,len(self.hideLayer))@self.hoWeights 
        return sigmoid(mulResultH2O-self.outputLayer).argsort()[0][-1]
        
    def saveToFile(self):
        self.ihWeights.tofile("ihWeights.bin")
        self.hoWeights.tofile("hoWeights.bin")
        self.hideLayer.tofile("hideLayer_bias.bin")
        self.outputLayer.tofile("outputLayer_bias.bin")

    def loadFromfile(self):
        self.ihWeights = (np.fromfile("ihWeights.bin",dtype=np.float64)).reshape(self.ihWeights.shape)
        self.hoWeights = (np.fromfile("hoWeights.bin",dtype=np.float64)).reshape(self.hoWeights.shape)
        self.hideLayer=(np.fromfile("hideLayer_bias.bin",dtype=np.float64))
        self.outputLayer=(np.fromfile("outputLayer_bias.bin",dtype=np.float64))

def getZipImagesAndTables(imgs,labels):
    return list(zip(imgs,labels))

def getOutputFormat(labels):
    fakeLabel = np.zeros((len(labels),10),dtype=np.uint8)
    for index in range(len(labels)):
        fakeLabel[index][labels[index]] = 1;
    return fakeLabel
