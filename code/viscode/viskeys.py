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

def annotate(frame):
	zf=0.15 #The zoom factor for the image2
	text1 = frame['gt_caption']
	path1 = frame['path']
	words = path1.split('/')
	sim1 = 'class : '+ words[len(words)-3] 
	knn = frame['knn']
	text2 = knn[0]['caption']
	bleu2 = "%.2f" %nltk.translate.bleu_score.sentence_bleu(text1, text2)
	path2 = knn[0]['path']
	words = path2.split('/')
	sim2 = 'class : '+ words[len(words)-3]+ ' Similarity : ' + str(knn[0]['similarity']) + ' BLEU :' + str(bleu2)
	text3 = knn[1]['caption']
	bleu3 = "%.2f" %nltk.translate.bleu_score.sentence_bleu(text1, text3)
        path3 = knn[1]['path']
	words = path3.split('/')
	sim3 = 'class : '+ words[len(words)-3] + ' Similarity: '+str(knn[1]['similarity']) + ' BLEU :' + str(bleu3)
	text4 = knn[2]['caption']
	bleu4 = "%.2f" %nltk.translate.bleu_score.sentence_bleu(text1, text4)
        path4 = knn[2]['path']
	words = path4.split('/')
	sim4 = 'class : '+ words[len(words)-3] + ' Similarity: '+ str(knn[2]['similarity']) + ' BLEU :' + str(bleu4)

	#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharey='row')

	# Define a 1st position to annotate (display it with a marker)
    	xy = (0.5, 0.7)
    	ax1.plot(xy[0], xy[1], ".r")

    	# Annotate the 1st position with a text box ('Test 1')
    	offsetbox = TextArea(text1, minimumdescent=False)

    	ab = AnnotationBbox(offsetbox, xy,
                        xybox=(0.5, 5.5),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"))
    	ax1.add_artist(ab)


	# Annotate the 1st position with a text box ('Test 1')
    	offsetbox = TextArea(sim1, minimumdescent=False)

    	ab = AnnotationBbox(offsetbox, xy,
                        xybox=(0.5, -5.5),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"))
    	ax1.add_artist(ab)

    	# Annotate the 1st position with another text box ('Test')
    	offsetbox = TextArea(text2, minimumdescent=False)

    	ab = AnnotationBbox(offsetbox, xy,
                        xybox=(0.5, 5.5),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"))
    	ax2.add_artist(ab)

	offsetbox = TextArea(sim2, minimumdescent=False)

    	ab = AnnotationBbox(offsetbox, xy,
                        xybox=(0.5, -5.5),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"))
    	ax2.add_artist(ab)


    	offsetbox = TextArea(text3, minimumdescent=False)

    	ab = AnnotationBbox(offsetbox, xy,
                        xybox=(0.5, 5.5),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"))
    	ax3.add_artist(ab)


	offsetbox = TextArea(sim3, minimumdescent=False)

    	ab = AnnotationBbox(offsetbox, xy,
                        xybox=(0.5, -5.5),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"))
    	ax3.add_artist(ab)


    	offsetbox = TextArea(text4, minimumdescent=False)

    	ab = AnnotationBbox(offsetbox, xy,
                        xybox=(0.5, 5.5),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"))

    	ax4.add_artist(ab)

	offsetbox = TextArea(sim4, minimumdescent=False)

    	ab = AnnotationBbox(offsetbox, xy,
                        xybox=(0.5, -5.5),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"))
    	ax4.add_artist(ab)



    	xy = (1, 0)

    	#arr = plt.imread(path1)
	arr = Image.open(path1)
	arr = arr.resize((480, 360), Image.ANTIALIAS)

    	im = OffsetImage(arr, zoom=zf)
    	im.image.axes = ax1

    	ab = AnnotationBbox(im, xy,
                        xybox=(0.5, 0),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"))

    	ax1.add_artist(ab)

	#Adding the second image now
    	#arr_img = plt.imread(path2)
	arr_img = Image.open(path2)
	arr_img = arr_img.resize((480, 360), Image.ANTIALIAS)

    	imagebox = OffsetImage(arr_img, zoom=zf)
    	imagebox.image.axes = ax2

    	ab = AnnotationBbox(imagebox, xy,
                        xybox=(0.5, 0),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"))

    	ax2.add_artist(ab)

    	#arr_img = plt.imread(path3)
	arr_img = Image.open(path3)
	arr_img = arr_img.resize((480, 360), Image.ANTIALIAS)

    	imagebox = OffsetImage(arr_img, zoom=zf)
    	imagebox.image.axes = ax2

    	ab = AnnotationBbox(imagebox, xy,
                        xybox=(0.5, 0),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"))

    	ax3.add_artist(ab)

    	#arr_img = plt.imread(path4)
	arr_img = Image.open(path4)
	arr_img = arr_img.resize((480, 360), Image.ANTIALIAS)

    	imagebox = OffsetImage(arr_img, zoom=zf)
    	imagebox.image.axes = ax2

    	ab = AnnotationBbox(imagebox, xy,
                        xybox=(0.5, 0),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"))

    	ax4.add_artist(ab)

    	# Fix the display limits to see everything
    	ax1.set_ylim(-5, 5)
    	ax1.set_xlim(-3, 3)
    	ax1.axis('off')
    	ax2.set_ylim(-5, 5)
    	ax2.set_xlim(-3, 3)
    	ax2.axis('off')
    	ax3.set_ylim(-5, 5)
    	ax3.set_xlim(-3, 3)
    	ax3.axis('off')
    	ax4.set_ylim(-5, 5)
    	ax4.set_xlim(-3, 3)
    	ax4.axis('off')
    	figure = plt.gcf() # get current figure
	names = path1.split('/')
	name = str(names[len(names)-2])+str(names[len(names)-1])
    	plt.savefig(name+'.png', bbox_inches='tight', dpi=600)
	plt.close()
	im = Image.open(name+'.png')
	os.remove(name+'.png') #Removing this file as we don't need it
	return im	
	

class viscaptions():
	def __init__(self, f):
		with open(f) as f1:
			self.data = json.load(f1)
		#Parsing the read data now
		self.parse()

	def parse(self):
		#This function will be used for parsing the read data
		for i in range(len(self.data)):
			self.parsevideo(self.data[i])

	def parsevideo(self, video):
		#This function parses the video
		self.name =  video['name']
		frames = video['frame_captions']
		#for i in range(len(frames)):
		#	frame = frames[i]
		#	im = annotate(frame)			
		#	print("Image read succesfully from", frame['path'])
		imframe = np.hstack([annotate(i) for i in frames])
		print(imframe.shape)
		imframe = np.hstack([imframe[:, 3201*i+550:3201*i+2500] for i in range(5)])
		img = Image.fromarray(imframe)
		img.save(self.name+'.png')
		print("completed the video of", self.name)

cap = viscaptions('/home/mscvproject/to_harsha/generate_caption_output.json')
			
		



