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

        # Assign each image a model, based on the sorted list
	imageToModel = np.zeros(numImages, dtype=int)
	for image in range(numImages):
		modelIndex = 0
		imageToModel[image] = -1
		for modelKey,_ in allObjectsSorted:
			currentModel = allObjects[modelKey]
			if image in currentModel:
				imageToModel[image] = modelIndex
				break
			modelIndex+=1

	unique, counts = np.unique(imageToModel, return_counts=True)
	print dict(zip(unique, counts))
	return imageToModel

# This method selects the models with the most images
def selectModelsMulti(numModels,MODEL_FILE,INVERT_INDEX_FILE,numImages):
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

        # Assign each image a model, based on the sorted list
	imageToModel = []
        averageLen = 0.0
	for image in range(numImages):
		imageList = set()
		modelIndex = 0
		for modelKey,_ in allObjectsSorted:
			currentModel = allObjects[modelKey]
			if image in currentModel:
				imageList.add(modelIndex)
			modelIndex+=1
		averageLen += len(imageList)
		if len(imageList) == 0:
			imageList=-1
		imageToModel.append(imageList)
	
	imageToModel = np.array(imageToModel)
	dictCount = {}
	for image in imageToModel:
		if image == -1:
			if -1 not in dictCount:
                                dictCount[-1] = 0
                        dictCount[-1] += 1
		else:
			for model in image:
				if model not in dictCount:
					dictCount[model] = 0
				dictCount[model] += 1
	print dictCount
	print 'Not Empty : '+str(averageLen/numImages)

	return imageToModel


if __name__ == '__main__':
    selectModelsMulti(100,'google.model','google.ifs',2000)

