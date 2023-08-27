import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib as mb
import io
ms=6 #marker size

# reading the Yeast data from the file "yeast.data"
def Class_readData(filename):
	data = []
	with io.open(filename, mode="r", encoding="utf-8") as f:
		#next(f) # incase we want to ignore a line
		for line in f:
			ldata = line.split()
			ldata.pop(0) #remove first column
			intData = []
			for l in range(len(ldata)):
				if l<len(ldata)-1:
					intData.append(float(ldata[l]))
				else:
					if ldata[l] == 'MIT':
						intData.append(1.0)
					elif ldata[l] == 'NUC':
						intData.append(2.0)
					elif ldata[l] == 'CYT':
						intData.append(3.0)
					elif ldata[l] == 'ME1':
						intData.append(4.0)
					elif ldata[l] == 'EXC':
						intData.append(5.0)
					elif ldata[l] == 'ME2':
						intData.append(6.0)
					elif ldata[l] == 'ME3':
						intData.append(7.0)
					elif ldata[l] == 'VAC':
						intData.append(8.0)
					elif ldata[l] == 'POX':
						intData.append(9.0)
					elif ldata[l] == 'ERL':
						intData.append(10.0)
					else:
						print("add ", ldata[l])
						break
			data.append(intData)
	
	return data


