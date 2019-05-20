import os
import csv
import pickle
from shutil import copyfile
import operator

inputpath = '../data/google-landmark-dataset-resize/'
outputpath = '../../google_landmark_selected/'
maxLandmarks = 40

groundTruthCount = {}
groundTruth = {}
with open('groundTruthLandCount.pickle', 'rb') as handle:
    groundTruthCount= pickle.load(handle)
with open('groundTruthLand.pickle', 'rb') as handle:
    groundTruth= pickle.load(handle)
print(len(groundTruth))

countLandmarks=0
for key,_ in sorted(groundTruthCount.items(), key=operator.itemgetter(1),reverse=True):
	print('Key : {} and size : {}'.format(key,len(groundTruth[key])))
	countImages = 0
	for value in groundTruth[key]:
		if not os.path.exists(outputpath+value+'.jpg'):
 			copyfile(inputpath+value+'.jpg', outputpath+value+'.jpg')
		countImages+=1
		if countImages >= 500:
			break
	countLandmarks+=1
	if countLandmarks == maxLandmarks:
		break
