#This function is to read the pca features present in /data01/mscvproject/data/ActivityNetCaptions/c3d
#folder.

import h5py
import os
import sys

class hdf5pca():
	def __init__(self, fp):
		self.fp =fp #The path to the file to read
	
	def read(self):
		f = h5py.File(self.fp, 'r')
		print("Keys: %s" % f.keys())
		a_group_key = list(f.keys())[0]
		data = (f[a_group_key])
		print("The data is", len(data))
		for k in data:
			print("The key is ", k)
			print("The value is ", data[k])

pca = hdf5pca('/data01/mscvproject/data/ActivityNetCaptions/c3d/PCA_activitynet_v1-3.hdf5')
pca.read()


