#This function reads all the features in the text file of the given folder and returns a dict
#The dict is of the type data['name'][num] where name is the name of the text file and the num is its
#frame number
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans
import sys
import os

class labelloader():
	def __init__(self, fpath):
		self.f=fpath #fpath is the folder
		self.data={}
		self.root = '/data01/mscvproject/data/ActivityNetCaptions/keyframes/training/'
	
	def returnkeys(self, k=3):
		#Iterates through all the text files and returns the features as a dict list
		self.k=k
		files = os.listdir(self.f)
		for f in files:
			ftext = self.root + f
			fabs = self.f +'/'+ f
			lines = [line.rstrip('\n') for line in open(fabs)]	
			key=f.split('.txt')[0]
			keys = self.extractkeykmeans(lines)
			keys = np.array(keys, dtype='int32')
			#keys = np.array(self.extractkeydist(lines), dtype='int')
			np.savetxt(ftext, keys)   # use exponential notation

	def extractkeykmeans(self, lines):
		#Reutrns the key frames based on kmeans clustering of the fc7 features
		keys=[]
		if len(lines)<self.k:
			keys=[x for x in range(len(lines))]
			return keys
		lines = [line.split(',') for line in lines]
		lines=np.array(lines, dtype='float')
		kmeans = KMeans(n_clusters=self.k, random_state=0).fit(lines)	
		for i in range(self.k):
			d = kmeans.transform(lines)[:, i]
			ind = np.argsort(d)[::][:1]
			keys.append(ind)
		return keys
		
	def extractkeydist(self, lines):
		#Returns the key frames from the fc7 features
		dist = np.array(self.calcdist(lines))
		keys=[]
		l = len(dist)
		print(l)
		inter = l/self.k
		for i in range(self.k):
			start=i*inter
			end=(i+1)*inter
			keys.append(start+np.argmax(dist[start:end]))
		return keys
		

	def calcdist(self, lines):
		#Computes the consecutive distance between the fc7 features
		dist=[]
		for i in range(len(lines)-1):
			currfeat = np.array(lines[i].split(','), dtype='float')
			nextfeat = np.array(lines[i+1].split(','), dtype='float')
			dist.append(norm(nextfeat-currfeat))
		return dist


class keyframeextractor():
	def __init__(self, fpath):
		self.f=fpath #fpath is the high level directory such as train, val, test containing sub directories
		self.data={}

	def extractframes(self, k=5):
		folders = os.listdir(self.f)
		num=0
		for f in folders:
			fabs = self.f +'/'+ f
			l=labelloader(fabs)
			l.returnkeys(k)
			num=num+1
			print("Completed this many folder key frame extraction", num)




