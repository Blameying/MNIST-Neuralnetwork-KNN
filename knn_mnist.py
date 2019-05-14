from mnistLoader import *
import numpy as np
from collections import Counter

def knn(test,trains,labels,k):
    if len(trains)!=len(labels):
        return
    length = len(trains)
    testArray = np.tile(test,(length,1))
    distances = (((testArray - trains)**2).sum(axis=1))**0.5
    result = list(distances.argsort())[:k]
    return  max(zip(*np.unique([labels[index] for index in result], return_counts=True)), key=lambda x: x[1])[0]
