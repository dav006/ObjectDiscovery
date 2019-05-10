import Image
import os

inputfolder = '/mnt/data/visual_instance_mining/annotations_landmarks_clean_validate/' 
outputfolder = '/mnt/data/visual_instance_mining/annotations_landmarks_clean_validate_crop/'

for file in sorted(os.listdir(inputfolder+'.')):
	if os.path.isfile(outputfolder+file):
		continue

	im = Image.open(inputfolder+file)
	width, height = im.size   # Get dimensions
       
        min_size = 0
        if width<height:
		min_size = width
	else:
		min_size = height
       
	left = (width - min_size)/2
	top = (height - min_size)/2
	right = (width + min_size)/2
	bottom = (height + min_size)/2

	imgNew = im.crop((left, top, right, bottom))
        imgNew.save(outputfolder+file+'.jpg')

