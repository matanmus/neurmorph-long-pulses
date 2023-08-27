import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib as mb
import matplotlib
from os.path import exists

#Script uses the Reservoir data to build the H matrix

valueList = np.arange(1.3, 2.1, 0.01)
fileNames = []
for val in valueList:
	fileNames.append('%.3f' % val)

ms=6 #marker size

def findIndices(Array):
	End=len(Array)
	stimI=np.argmin(np.abs(Array-x0))
	A=Array[stimI:End]
	minI=np.argmin(A)
	maxI=np.argmax(A)
	val=(A[maxI]+A[minI])/2
	rightI=stimI+maxI+np.argmin(np.abs(A[maxI:len(A)]-val))
	leftI=stimI+np.argmin(np.abs(A[0:maxI+1]-val))
	return (stimI+minI, stimI+maxI, rightI, leftI)

matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rcParams.update({'font.size': 28})

y_output = [-3, -2, -1, 0, 1, 2, 3] #output position
tMin_output = 0.11 #output time
tMax_output = 0.29 #output time

linestyles = ['-', ':', '--', '-.', '-', ':', '--',  '-.', '-', ':', '--', '-.', '-', ':']
for k in fileNames:
	name = k
	name+= '_data.npz'
	data = np.load(name)
	# parameters
	value = data['value']
	xInput = data['xInput']
	A = data['A']
	w0 = data['w0']
	dom = data['dom']
	N = data['N']
	dt = data['dt']
	iter = data['iter']
	t_plots=data['t_plots']
	t_array=data['t_array']
	rho_array=data['rho_array']
	p_array=data['p_array']
	T_array=data['T_array']
	w_array=1/rho_array

	dx = 2 * dom / N
	XaxisEuler = np.cumsum(w_array, axis=1)*dx/w0 - dom

	# # # Calculate max output from a time range
	indexLow = int(tMin_output/(dt*iter))
	indexHigh = int(tMax_output/(dt*iter))
	index = np.arange(indexLow, indexHigh, 1)
	outputRhoMax = []
	outputPiMax = []
	for l in range(len(y_output)): #go on every y_output
		outputRho = []
		outputPi = []
		for i in index:
			idx = np.argmin(np.abs(XaxisEuler[i] - y_output[l]))
			outputRho.append(rho_array[i, idx])
			outputPi.append(p_array[i, idx])
		outputRhoMax.append(max(outputRho))
		outputPiMax.append(max(outputPi))

	if exists('H.npz'):
		Hdata = np.load('H.npz')
		Hrho = Hdata['Hrho']
		Hpi = Hdata['Hpi']
		Alist = Hdata['Alist']
	else:
		Alist = np.zeros((1, 1))
		Hrho = np.zeros((len(y_output), 1))
		Hpi = np.zeros((len(y_output), 1))
	AlistNew = np.c_[Alist, value]
	HrhoNew = np.c_[Hrho, outputRhoMax]
	HpiNew = np.c_[Hpi, outputPiMax]

	np.savez('H.npz', Hrho=HrhoNew, Hpi = HpiNew, Alist = AlistNew)
print(AlistNew)


