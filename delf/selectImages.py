import smh
from collections import Counter
import pickle
import operator
from config import Config

model = smh.listdb_load(Config.MODEL_FILE)
ifs = smh.listdb_load(Config.INVERT_INDEX_FILE)
idToFileName = ()
with open('indexToFile.pickle', 'rb') as handle:
    idToFileName= pickle.load(handle)

groundTruthFile = 'googleGroundTruth_selected.pickle'
groundTruthImages = {}
with open(groundTruthFile, 'rb') as handle:
    groundTruthImages= pickle.load(handle)


# Create array with all images associated with a objectDiscovered
objectDiscovered = model.ldb[32]
imageCount={}
for visualWord in objectDiscovered:
	visualWordName = visualWord.item
        for image in ifs.ldb[visualWordName]:
		if image.item not in imageCount:
			imageCount[image.item] = 0
                imageCount[image.item] += 1

sort = sorted(imageCount.items(), key=lambda x: x[1], reverse=True)
fileName = idToFileName[sort[0][0]]
print(fileName)
print(fileName in groundTruthImages['10900'])
