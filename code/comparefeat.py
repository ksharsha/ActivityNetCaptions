#This function compares a given feature against the training features to retrieve the k nearest neighbors
import os
import numpy as np
from featureread import featloader
import pickle

class knn_feat():
	def __init__(self, feat=None, folder=None):
		self.feat = feat #The feature to be compared
		self.folder = folder
		with open('trainfeats.pickle', 'rb') as handle:
			self.feats = pickle.load(handle)

	def knn(self):
		your_data = {'foo': 'bar'}
		#self.featl = featloader(self.folder)
		#self.feats = self.featl.returnfeats()
		#with open('trainfeatsnew.pickle', 'wb') as handle:
		#	pickle.dump(self.feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

		return self.feats
