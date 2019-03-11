import sklearn.metrics

def computeOxfordAP(pos, amb, ranked_list):
	old_recall = 0.0;
	old_precision = 1.0;
	ap = 0.0;

	intersect_size = 0;
	i = 0;
	j = 0;
	for i in range(len(ranked_list)):
		if ranked_list[i] in amb:
			continue
		elif ranked_list[i] in pos:
			intersect_size+=1

		recall = intersect_size / float(len(pos))
		precision = intersect_size / (j + 1.0)

		ap += (recall - old_recall)*((old_precision + precision)/2.0)

		old_recall = recall
		old_precision = precision
		j+=1

	return ap


def computeMyAP(positive, junk, ranked_list):
	precision= []
	recall = []
	
	precision.append(1.0)
	recall.append(0.0)
	truePositives = 0.0
	retrievedImages = 0.0

	positiveImages = float(len(positive))

	for fileOb in ranked_list:
		if fileOb in junk:
			continue
		elif fileOb in positive:
			truePositives +=1.0
		
		retrievedImages+=1.0
		precision.append(truePositives/retrievedImages)
		recall.append(truePositives/positiveImages)

	ap = sklearn.metrics.auc(recall,precision)
	return ap

def computeMyAPGoogle(positive, ranked_list):
        precision= []
        recall = []

        precision.append(1.0)
        recall.append(0.0)
        truePositives = 0.0
        retrievedImages = 0.0

        positiveImages = float(len(positive))

        for fileOb in ranked_list:
                if fileOb in positive:
                        truePositives +=1.0

                retrievedImages+=1.0
                precision.append(truePositives/retrievedImages)
                recall.append(truePositives/positiveImages)

        ap = sklearn.metrics.auc(recall,precision)
        return ap
