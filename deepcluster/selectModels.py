import smh
import pickle
import operator
import numpy as np

# This method selects the models with the most images
def selectModels(numModels,MODEL_FILE,INVERT_INDEX_FILE,numImages):
	model = smh.listdb_load(MODEL_FILE)
	ifs = smh.listdb_load(INVERT_INDEX_FILE)

        # For each model count the number of times an image appears, also count the numero of images that appear on a model
    	allObjects = {}
        allObjectsCountImages = {}
        index = 0
	for objectDiscovered in model.ldb:
		imageCount={}
                count=0
		for visualWord in objectDiscovered:
			visualWordName = visualWord.item
			for image in ifs.ldb[visualWordName]:
				if image.item not in imageCount:
					imageCount[image.item] = 0
					count+=1
				imageCount[image.item] += 1
				#count+=1
		allObjects[index]=imageCount
		allObjectsCountImages[index]=count
		index+=1

	# Sort models based on the number of images in each model
	allObjectsSorted = sorted(allObjectsCountImages.items(), key=lambda x: x[1], reverse=True)[0:numModels]

	imageToModel = np.zeros(numImages, dtype=int)
	imageToMax = np.zeros(numImages, dtype=int)
	for image in range(numImages):
		for modelKey,_ in allObjectsSorted:
			currentModel = allObjects[modelKey]
			if image in currentModel and currentModel[image] > imageToMax[image]:
				imageToMax[image] = allObjects[modelKey][image]
				imageToModel[image] = modelKey

	mappedImageToModel = np.zeros(numImages, dtype=int)
	for image in range(numImages):
		for model in range(numModels):
			if imageToModel[image] == allObjectsSorted[model][0]:
				mappedImageToModel[image]=model
				break
	'''
	for image in range(numImages):
		print str(imageToModel[image])+':'+str(imageToMax[image])
	'''

	unique, counts = np.unique(imageToModel, return_counts=True)
	print dict(zip(unique, counts))
	return mappedImageToModel
'''
if __name__ == '__main__':
    selectModels(40,'google.model','google.ifs',500)
'''
