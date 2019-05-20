import os
import pickle

index = 0
dictIndexFile = {}
for file in sorted(os.listdir('../../google_landmark_selected_crop/subfolder/'+'.')):
	removedExt = ".".join(file.split(".")[:-1])
	dictIndexFile[index] = removedExt
	index+=1

with open('indexToFile.pickle', 'wb') as handle:
    pickle.dump(dictIndexFile, handle, protocol=pickle.HIGHEST_PROTOCOL)
