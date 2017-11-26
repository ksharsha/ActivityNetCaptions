#This function reads all the features in the text file of the given folder and returns a dict
#The dict is of the type data['name'][num] where name is the name of the text file and the num is its
#frame number
import numpy as np
import sys
import os

class folderloader():
	def __init__(self, fpath):
		self.f=fpath #fpath is the folder
		self.data={}
	
	def returndict(self):
		#Iterates through all the text files and returns the features as a dict list
		files = os.listdir(self.f)
		for f in files:
			fabs = self.f +'/'+ f
			lines = [line.rstrip('\n') for line in open(fabs)]	
			key=f.split('.txt')[0]
			self.data[key]=lines
		return self.data

class featloader():
	def __init__(self, fpath):
		self.f=fpath #fpath is the high level directory such as train, val, test containing sub directories
		self.data={}

	def returnfeats(self):
		folders = os.listdir(self.f)
		num=0
		for f in folders:
			fabs = self.f +'/'+ f
			l=folderloader(fabs)
			d=l.returndict()
			self.data[f]=d
			num=num+1
			print("Completed these many folders", num)
		return self.data
		
