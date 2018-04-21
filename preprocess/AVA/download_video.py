# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import json
import os
import csv
videoList = []
with open("./ava_train_v1.0.csv") as f1:
    reader = csv.reader(f1, delimiter=',')
    for row in reader:
        videoList.append(row[0])
# annotation_file = open('activity_net.v1-3.min.json')
# annotation = json.load(annotation_file)

# video_database = annotation['database']
# videos = annotation['database'].keys()
videoList = list(set(videoList))
print(len(videoList))
# Download the ActivityNet videos into the ./videos folder
command1 = 'mkdir '+'videos'
os.system(command1)

for i in videoList:
    # url = video_database[i]['url']
    command3 = 'youtube-dl -o '+'videos/'+i+' ' + "https://www.youtube.com/watch?v="+i + ' -f worst'
    print command3
    os.system(command3)



