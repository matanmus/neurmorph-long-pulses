import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib as mb
import matplotlib
from os.path import exists
from sklearn import utils

# Script loads the H matrix, removes one data point from the H matrix and tests the quality of prediction using the H matrix on the data point

ms=6 #marker size

matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rcParams.update({'font.size': 28})

linestyles = ['-', ':', '--', '-.', '-', ':', '--',  '-.', '-', ':', '--', '-.', '-', ':']

Hdata = np.load('H.npz')
Hrho = Hdata['Hrho']
Hpi = Hdata['Hpi']
Alist = Hdata['Alist']

#shuffle data to extract new test every use of script
indices = list(range(0, len(Alist[0]), 1))
print(len(indices))
testXlist = []
resRhoList = []
resPiList = []
for j in indices:
	inds = indices.copy()
	inds.remove(j)
	Nx = Hrho.shape[0]
	Ny = Hrho.shape[1]
	trainRho = Hrho[:, inds]
	testRho = Hrho[:, j]
	trainPi = Hpi[:, inds]
	testPi = Hpi[:, j]
	trainX = Alist[0][inds]
	testX = Alist[0][j]
	trainX = trainX.reshape((Ny-1,1))

	invHrho = np.linalg.pinv(trainRho)
	invHpi = np.linalg.pinv(trainPi)
	betaRho = np.dot(np.transpose(invHrho), np.sin(trainX))
	betaPi = np.dot(np.transpose(invHpi), np.sin(trainX))

	#test
	sol = np.sin(testX)
	if sol != 0:
		resRho = abs(np.dot(np.transpose(testRho), betaRho) - sol)/sol
		resRho = resRho[0]
		resPi = abs(np.dot(np.transpose(testPi), betaPi) - sol)/sol
		resPi = resPi[0]
	else:
		resRho = float("nan")
		resPi = float("nan")
	testXlist.append(testX)
	resRhoList.append(resRho)
	resPiList.append(resPi)

np.savez('prediction.npz', testXlist=testXlist, resRhoList = resRhoList, resPiList = resPiList)
