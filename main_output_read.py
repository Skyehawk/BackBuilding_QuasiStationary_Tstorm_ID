import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates 

from metpy.plots import ctables


if __name__ == "__main__":
	objects = []
	with (open("test.txt.p", "rb")) as openfile:
		while True:
			try:
				objects.append(pickle.load(openfile))
			except EOFError:
				break

	obj = objects[0]
	obj = obj.transpose(2, 1, 0)	#transpose our 3d array for more logical slicing

	xlocs = obj[:][:][0].astype(float)
	print(f'xlocs shape: {np.shape(xlocs)}')
	#print(xlocs[0][0])

	strtimes = obj[:][:][1]
	dataDatetimes = [np.datetime64(datetime.strptime(t,'%Y-%m-%dT%H:%M:%SZ')) for t in strtimes.ravel()]
	#dataDatetimes = [datetime.strptime(t,'%Y-%m-%dT%H:%M:%SZ') for t in strtimes.ravel()]
	dataDatetimes = np.reshape(dataDatetimes, np.shape(strtimes))
	print(f'datetime shape: {np.shape(dataDatetimes)}')
	#print(dataDatetimes)
	#print(type(dataDatetimes[0][0]))

	data  = obj[:][:][2].astype(float)
	print(f'data shape: {np.shape(data)}')
	#print(data[np.shape(data)[0]//2:])						# There are a lot of nans near edges, so print may not be super insightful, so we index to grab a chunk out of the middle
	#print(type(data[0][0]))

	fig, axes = plt.subplots(2,2, figsize=(20,20),
        gridspec_kw={'width_ratios':[10, 10], 
     	'height_ratios': [10, 10], 'wspace': 0.375,
     	'hspace': 0.375})

	date_format = mpl_dates.DateFormatter('%H:%Mz')
	norm, cmap = ctables.registry.get_with_steps('NWSReflectivity', 5,5)

	axes[0][0].pcolormesh( xlocs, dataDatetimes, data, norm=norm, cmap=cmap, shading='auto')
    
	axes[0][0].set_xlabel("Degrees Longitude From Initiation Point")
	axes[0][0].set_ylabel("Time (UTC)")
	axes[0][0].set_title("Hovmöller Diagram - Mean Reflectivity")

	plt.show()
	