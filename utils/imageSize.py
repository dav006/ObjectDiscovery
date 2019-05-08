import Image
import os

folder = '/mnt/data/visual_instance_mining/ObjectDiscovery/data/google-landmark-dataset/' 

minWidth = 2000
minHeight = 2000

count=0
for file in sorted(os.listdir(folder+'.')):
	im = Image.open(folder+file)
	width, height = im.size
	if(width < minWidth):
		minWidth = width
	elif(height < minHeight):
                minHeight = height
	count+=1
	print(count)

print('Min width : {}  Min height : {}'.format(minWidth,minHeight))	
