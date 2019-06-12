import sys
import os
from config import Config
import numpy as np
import hnswlib
sys.path.append('../common/')
import dumpy


stopWords = set()

DIM = 128
VISUAL_VOCAB = 1000000
inputpath = '/mnt/data/visual_instance_mining/oxford5k_LIFT/'


# Reiniting, loading the index
p = hnswlib.Index(space='l2', dim=DIM)
p.load_index("1000000/hsm_1000000_lift_30iter.bin", max_elements = VISUAL_VOCAB)

#Convert DELF descriptors to visual words for each delf file
print('Convert LIFT')
filelist = sorted(os.listdir(inputpath+'.'))
allVisualWords = []
count = 0
for entry in filelist:
	# Read features
        fileObj = dumpy.loadh5(inputpath+entry)
        fileObj = fileObj['descriptors']
        size = len(fileObj)
	allDesc = fileObj
        labels, _ = p.knn_query(allDesc, k=1)
	npArray = labels.flatten()
	visualwords = set(npArray)
	allVisualWords.append(visualwords)
	count+=1
	print(count)

print('Get global count')
# Get global count
globalCount = {}
for wordToFreq in allVisualWords:
	for wordId in wordToFreq:
		if wordId not in globalCount:
			globalCount[wordId] = 1
		else:
			globalCount[wordId] += 1

minCount = 5
maxCount = 1519
#print('Min frecuency : {} and Max frecuency : {}'.format(minFreq,maxFreq))
addStop = 0
for key,value in globalCount.items():
	if value < minCount:
		stopWords.add(key)
		addStop+=1
	elif value > maxCount:
		stopWords.add(key)
		addStop+=1
print('Stop words : {}'.format(addStop))

countVis = 0
for wordToFreq in allVisualWords:
	countVis+=len(wordToFreq)
print('Visual words before stop words : {}'.format(countVis))

#Remove stop words
print('Remove stop words')
allWordToFreqNoStop = []
for wordToFreq in allVisualWords:
	allWordToFreqNoStop.append(wordToFreq.difference(stopWords))

countVis = 0
for wordToFreq in allWordToFreqNoStop:
	countVis+=len(wordToFreq)
print('Visual words after stop words : {}'.format(countVis))

#Remove stop words
print('Save binary BoW without stopWords')
outputFile = open(Config.CORPUS_FILE,'w')
for wordToFreq in allWordToFreqNoStop:
	#Write the dictionary in the default format
	outputFile.write(str(len(wordToFreq)))
	for wordId in wordToFreq:
		outputFile.write(' ')
		outputFile.write(str(wordId)+':1')
	outputFile.write('\n')
outputFile.close
