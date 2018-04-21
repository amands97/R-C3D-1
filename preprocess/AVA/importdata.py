import csv
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def importData(split):
	filename = "ava_" + split.lower() + "_v1.0.csv"
	dataDict_ = {}
	num = 0
	a = []
	for i in range(1,201):
		a.append(i)
	with open(filename) as f:
		reader_ = csv.reader(f, delimiter = ',')
		for row in reader_:
			vidName = row[0]
			if vidName not in dataDict_:
				dataDict_[vidName] = {}
				# print(vidName)
				# num += 1
			
			label = int(row[6])
			# print(label)
			if vidName == "-5KQ66BBWC4" and row[1] == "0904":
				print(label)
			if float(row[1]) in dataDict_[vidName]:
				dataDict_[vidName][float(row[1])].append(label)
			else:
				dataDict_[vidName][float(row[1])] = [label]
		for vid in dataDict_:
			# print(vid)
			for time in dataDict_[vid]:
				# if vid == "-5KQ66BBWC4" and time == 904:
				# print(dataDict_[vid][time])
				mlb = MultiLabelBinarizer(a)
				# label = int(row[6])
				# print(label)
				labels = []
				# labels.append(label)
				# print(labels)
				for label in dataDict_[vid][time]:
					labels.append(tuple([label]))

				# t = tuple(labels)
				# print(t)
				# b = []
				# b.append(t)
				# print(labels)
				temp1 = np.sum(mlb.fit_transform(labels).tolist(), axis = 0)
				# print(temp1)
				temp =[temp1>1]
				# if vid == "-FaXLcSFjUI" and time == 1156:
					# print(temp1, temp)
				# temporary = np.sum(mlb.fit_transform(labels).tolist(), axis = 0)
				# print(temp)
				temp1[temp] = 1
				# for i, element in enumerate(temporary):
					# if element > 1:
						# temporary[i] = 1
				dataDict_[vid][time] = temp1
	# return dataDict_
			
	# print(num)

	dataDict_ = importData("train")
	# print(dataDict_["-FaXLcSFjUI"][1156])
	# print(dataDict_)
	segment1 = {}
	segment2 = {}
	for element in dataDict_:
		segment1[element] = np.column_stack(([x - 1.5 for x in dataDict_[element].keys()], [y+1.5 for y in dataDict_[element].keys()])) 
		segment2[element] = dataDict_[element].values()

	print(segment2["-FaXLcSFjUI"][12], segment1["-FaXLcSFjUI"][12])
	return segment1, segment2