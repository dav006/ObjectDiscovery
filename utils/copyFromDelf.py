import os
import csv
import pickle
from shutil import copyfile
import operator

#inputpath = '../../google_landmark_attention/'
inputpath = '../../google_landmark_selected/'
delfpath = '../../google_landmark_attention/'
outputpath = '../../google_landmark_attention_selected/'

files = [None]*208463

index=0
print('Reading Files')
for file in os.listdir(inputpath+'.'):
        removedExt = ".".join(file.split(".")[:-1])
        files[index] = removedExt
        index+=1

for file in files:
	if file is not None and os.path.exists(delfpath+file+'.delf') and not os.path.exists(outputpath+file+'.delf'):
 		copyfile(delfpath+file+'.delf', outputpath+file+'.delf')
		print(file)
