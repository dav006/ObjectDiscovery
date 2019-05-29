import os
import numpy as np
import hnswlib
import sys

def createCorpus(X_data_conv,CORPUS_FILE):
	stopWords = set()
	inputpath = '../../google_landmark_attention_selected/'

	# Reiniting, loading the index
	p = hnswlib.Index(space='l2', dim=40)
	p.load_index("hsm_25000_30iter_google_40_balan.bin", max_elements = 25000)

	#Convert Conv descriptors to visual words
	print('Convert Conv')
	allVisualWords = [None]*X_data_conv.shape[0]
	count = 0
	for entry in X_data_conv:
        	labels, _ = p.knn_query(entry, k=1)
		npArray = labels.flatten()
		visualwords = set(npArray)
		allVisualWords[count] = visualwords
		count+=1
		#print(count)
		#sys.stdout.flush()

	print('Get global count')
	# Get global count
	globalCount = {}
	for wordToFreq in allVisualWords:
		for wordId in wordToFreq:
			if wordId not in globalCount:
				globalCount[wordId] = 1
			else:
				globalCount[wordId] += 1
	print('Global count : {}'.format(len(globalCount)))
	minCount = 20
	maxCount = 6000

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
	outputFile = open(CORPUS_FILE,'w')
	for wordToFreq in allWordToFreqNoStop:
		#Write the dictionary in the default format
		outputFile.write(str(len(wordToFreq)))
		for wordId in wordToFreq:
			outputFile.write(' ')
			outputFile.write(str(wordId)+':1')
		outputFile.write('\n')
	outputFile.close
