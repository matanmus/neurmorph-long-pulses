import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import exists

# Script uses the Reservoir data to build the H matrix 

def extractData(k, yTrain):
	y_output = np.arange(-9.5, 9.5, 19/100) #output position
	#print(len(y_output))
	#print(y_output)
	tMin_output = 0.11 #output time
	tMax_output = 0.29 #output time

	name = 'data/'
	name += str(k)
	name+= '_data.npz'
	data = np.load(name)
	# parameters
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
	y = np.zeros((10, 1))
	y[int(yTrain - 1)] = 1
	if exists('H.npz'):
		Hdata = np.load('H.npz')
		Hrho = Hdata['Hrho']
		Hpi = Hdata['Hpi']
		ylist = Hdata['ylist']
		ylistNew = np.c_[ylist, y]
	else:
		Hrho = np.zeros((len(y_output), 1))
		Hpi = np.zeros((len(y_output), 1))
		ylistNew = list(y)
	HrhoNew = np.c_[Hrho, outputRhoMax]
	HpiNew = np.c_[Hpi, outputPiMax]

	np.savez('H.npz', Hrho=HrhoNew, Hpi = HpiNew, ylist = ylistNew)



