import os
import fnmatch
import sys


#Opening the file to write
text_file = open("c3doutp.txt", "w")
def countframes(filelist):
	rgb_prefix = '0'
	prefix_list = (rgb_prefix)
	cnt_list=[len(fnmatch.filter(filelist, x+'*')) for x in prefix_list]
	return cnt_list

f=open("/data01/mscvproject/code/temporal-segment-networks/data/ActNet_splits/ActNetClasses.txt")

classes=f.read()
classeslist=classes.split("\n")
#cdict will be a dictionary containing the mapping from the class name to its index
cdict={}
for i in range(0,200):
	d=classeslist[i]
	e=d.split(":")
	str1=(str(e[0]))
	cdict[str1]=int(e[1])


#We will now go into all the directories and create the split file
classnames = [name for name in os.listdir('.') if os.path.isdir(name)]
print(classnames)
for i in range(0,len(classnames)):
	print('In the directory number',i)
	index = cdict[classnames[i]]
	substr = "./" + str(classnames[i])
	#print('The sub folder is',substr)
	for root, dirs, files in os.walk(substr):
		subroot=root.split("/")
		#We want the directories inside the labels only
		if(len(subroot)==3):
			#print(subroot)
			rootpath=str(os.path.dirname(os.path.abspath(root))) + "/" +  subroot[2]
			filepath = substr + "/" +  subroot[2]
			#print(rootpath)
			#print(filepath)
			for rootsub, dirsub, filesub in os.walk(filepath):
				#Uncomment the below ones too rename
				#for i in filesub:
				#	oldpath = rootpath+'/' + i
				#	newpath = rootpath +'/'+'0'+i
				#	os.rename(oldpath, newpath)
				#	print(oldpath, newpath)
				counts=countframes(filesub)
				counts = counts[0]
				print (counts)
				#Taking minimum just to be on the safe side, ideally they should be equal
				num = 1
				while counts>15 :
					#finalpath = rootpath + '/' +' '+ str(num)+' '+ '0' #Input
					finalpath = rootpath + '/'+str(num) #Output 
					text_file.write("{}\n" .format(finalpath))
					counts = counts - 1
					num = num + 1



text_file.close()
	#print('The number of subdirectories inside this folder is',len(subclasses))
