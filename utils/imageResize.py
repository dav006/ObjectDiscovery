import Image
import os

folder = '/mnt/data/visual_instance_mining/ObjectDiscovery/data/google-landmark-dataset/' 
outFolder = '/mnt/data/visual_instance_mining/ObjectDiscovery/data/google-landmark-dataset-resize/' 

minLen = 256.0
maxLen = 2048.0

count=0
countTotal = 0
for file in sorted(os.listdir(folder+'.')):
	if os.path.isfile(outFolder+file):
		countTotal+=1
		continue
	im = Image.open(folder+file)
	width, height = im.size
	rescaleDown = False
	if(width > maxLen or height > maxLen):
		rescaleDown = True

	if(rescaleDown):
		count+=1
		newWidth = 0
		newHeight = 0
		newLen = 0
		if rescaleDown:
			newLen = maxLen
		
		if (width < height ):
			ratio = newLen/float(height)
			newWidth = width*ratio
			newHeight = newLen
		else:
			ratio = newLen/float(width)
                        newHeight= height*ratio
                        newWidth = newLen
			
		imgNew = im.resize((int(newWidth),int(newHeight)), Image.ANTIALIAS)
		imgNew.save(outFolder+file)
		print('Resized {}'.format(count)) 
	else:
		im.save(outFolder+file)
	countTotal +=1
	print('Total {}'.format(countTotal))
