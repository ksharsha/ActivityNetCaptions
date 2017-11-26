#This function will be used to visualize the key images and their corresponding captions using the KNN appraoch

import os
import sys
import json
import nltk
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 3}

matplotlib.rc('font', **font) #Setting the font to low to begin with

class viscaptions():
	def __init__(self, f):
		with open(f) as f1:
			self.data = json.load(f1)
		#Parsing the read data now
		with open('/data01/mscvproject/data/ActivityNetCaptions/.scripts/valdump.json') as f:
			data = json.load(f)
		self.valmap = data['valmap']
		self.valtimes = data['valtimes']
		self.valdata = {}
		self.valdata['version'] = 'VERSION 1.0'
		self.valdata['external_data'] = 'TimePass'
		self.res = {}
		self.parse()

	def parse(self):
		#This function will be used for parsing the read data
		for i in range(len(self.data)):
			self.parsevideo(self.data[i])
		self.valdata['results'] = self.res
		with open('val_submission.json', 'w') as f:
			json.dump(self.valdata, f)

	def parsevideo(self, video):
		#This function parses the video
		self.name =  video['name']
		frames = video['frame_captions']
		sam = {}
		sam['sentence'] = self.aggcap(frames)
		sam['timestamp'] = self.valtimes[self.name]
		vid = self.valmap[self.name]
		if vid not in self.res:
			self.res[vid] = []
			self.res[vid].append(sam)
		else:
			sams = self.res[vid]
			sams.append(sam)
			self.res[vid] = sams

	def aggcap(self, frames):
		#This function aggregates the frame captions
		scores = []
		sim = 0 
		for i in range(len(frames)):
			frame = frames[i]['knn'][0]
			if 'similarity' in frame:
				scores.append(frame['similarity'])
				sim = 1 
			else:
				scores.append(frame['distance'])
		scores = np.array(scores)
		if sim == 1:
			ind = np.argmax(scores)
		else:
			ind = np.argmin(scores)
		caption = frames[ind]['knn'][0]['caption']
		return caption
		


cap = viscaptions('/home/mscvproject/to_harsha/generate_caption_output.json')
		

