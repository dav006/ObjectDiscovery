import os

# Get all images that belong to a landmark (groundtruth)
# E.g :all souls [all_souls_000091, all_souls_000026, oxford_003410, ..]
def groundTruthLoader(groundTruthPath):

	okAndGoodGroundTruthImages = {}
	junkGroundTruthImages = {}

	for file in sorted(os.listdir(groundTruthPath)):
	
		f = open(groundTruthPath+file,'r')
		splits = file.split("_")
		groundTruthName = splits[0]

		# Check if is junk file
		extensionName = splits[len(splits)-1]
		
		if extensionName == 'junk.txt':
			if groundTruthName not in junkGroundTruthImages:
				junkGroundTruthImages[groundTruthName] = []
			junkGroundTruthArray = junkGroundTruthImages[groundTruthName]

			for line in f:
				fileName = line.rstrip()
				if fileName not in junkGroundTruthArray:
					junkGroundTruthArray.append(fileName)
		elif extensionName == 'good.txt' or extensionName == 'ok.txt':
			
			if groundTruthName not in okAndGoodGroundTruthImages:
				okAndGoodGroundTruthImages[groundTruthName] = []
			groundTruthArray = okAndGoodGroundTruthImages[groundTruthName]

			for line in f:
				fileName = line.rstrip()
				if fileName not in groundTruthArray:
					groundTruthArray.append(fileName)
		f.close
	return okAndGoodGroundTruthImages,junkGroundTruthImages