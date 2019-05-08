import os
import pickle
from config import Config

index = 0
dictIndexFile = {}
for file in sorted(os.listdir('../../google_landmark_attention_selected/'+'.')):
	removedExt = ".".join(file.split(".")[:-1])
	dictIndexFile[index] = removedExt
	index+=1

with open('indexToFile.pickle', 'wb') as handle:
    pickle.dump(dictIndexFile, handle, protocol=pickle.HIGHEST_PROTOCOL)
