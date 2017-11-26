#This file will be used to create teh hdf5 file that will be given as input to the video action proposal framework
import os
import sys
import h5py
import numpy as np
from sklearn.decomposition import PCA

def read_binary_blob (file_name):
    
    fid = open(file_name, 'r')
    
    #s contains size of the blob e.g. num x chanel x length x height x width
    s = np.fromfile(fid, np.int32, 5)

    m = s[0] * s[1] * s[2] * s[3] * s[4]

    # data is the blob binary data in single precision (e.g float in C++) 
    data = np.fromfile(fid, np.float32, m)

    fid.close()
    
    return (s, data)

class createh5py():
	def __init__(self, path):
		data = [line.rstrip('\n') for line in open(path)]
		self.paths = [str(x)+'.fc7-1' for x in data]
		
	def create(self):
		print("The data length is", len(self.paths))
		currvid = []
		feats = {} #This is a dictionary with video name and their PCA features stored accordingly
		num = 0
		for path in self.paths:
			words = path.split('/')
			currvid = words[len(words)-2]
			_, feat = read_binary_blob(path)
			feat = np.expand_dims(feat, axis=0)
			#if num==0:
			#	feats['all'] = feat
			#else:
			#	allf = feats['all']
			#	feats['all'] = np.vstack((allf, feat))
			if currvid in feats:
				currfeats = feats[currvid]
				currfeats = np.vstack((currfeats, feat))	
				feats[currvid] = currfeats
			else:
				feats[currvid] = feat
			print("The num is", num)
			num = num + 1
		
		allfeats = np.vstack([feats[vid] for vid in feats])	
		pca = PCA(n_components=500)
		print("The input shape of features is", allfeats.shape)
		pca_feat = pca.fit_transform(allfeats)
		print("The pca feat of all the videos is of shape", pca_feat.shape)
		index = 0
		feats = {} #Clearing the previous feature vectors
		pcafeats = {}
		for path in self.paths:
			words = path.split('/')
			currvid = words[len(words)-2]
			feat = pca_feat[index,:]
			if currvid in pcafeats:
				currfeats = pcafeats[currvid]
				currfeats = np.vstack((currfeats, feat))
				pcafeats[currvid] = currfeats
			else:
				pcafeats[currvid] = feat
			index = index+1

		h = h5py.File('ActNet_c3d.hdf5', 'w')
		for k in pcafeats:
			grp = h.create_group(k)
			v = pcafeats[k]
			grp.create_dataset('c3d_features', data=v)
		

f = createh5py('/home/mscvproject/VideoCaptioning/harsha/c3d/newC3D/C3D/C3D-v1.0/examples/c3d_feature_extraction/prototxt/activity_list_out.txt')
f.create()
