import os
import pickle
from config import Config

index = 0
dictIndexFile = {}
for file in sorted(os.listdir('/mnt/data/visual_instance_mining/oxbuild_images/'+'.')):
        removedExt = ".".join(file.split(".")[:-3])
        dictIndexFile[index] = removedExt
        index+=1

with open('indexToFile.pickle', 'wb') as handle:
    pickle.dump(dictIndexFile, handle, protocol=pickle.HIGHEST_PROTOCOL)
