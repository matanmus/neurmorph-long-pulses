import numpy as np
import matplotlib.pyplot as plt
from readClassificationData import readClassificationData
from stimSeq import stimSeq
from extractDataOnlyTwo import extractDataOnlyTwo
import time

# Main script: loads the yeast data file, runs it through the reservoir and generates the H matrix

filename = "yeast.data"
dataSet = Class_readData(filename)

start_time = time.time()

for k in range(1, range(len(dataSet)):
    if k % 100 == 0:
        print("data set number: ", k)
    data = dataSet[k]
    y = data[-1]
    A = data[:-1]
    Class_reservoir(A, k) #Stimulate the reservoir to generates output data
    Class_extractData(k, y) # Script uses the Reservoir data to build the H matrix

print("--- %s seconds ---" % (time.time() - start_time))