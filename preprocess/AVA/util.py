# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------
from operator import itemgetter
import subprocess
import shutil
import os, errno
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
def generate_classes(data):
  class_list = []
  for vid, vinfo in data['database'].iteritems():
    for item in vinfo['annotations']:
      class_list.append(item['label'])

  class_list = list(set(class_list))
  classes = {'Background': 0}
  for i,cls in enumerate(class_list):
     classes[cls] = i + 1
  return classes


def generate_segment(split, data, classes):
  segment1 = {}
  segment2 = {}
  segment3 = {}
  segment = {}
  VIDEO_PATH = 'frames/%s/' % split
  video_list = set(os.listdir(VIDEO_PATH))
  # get time windows based on video key
  a = []
  for i in range(1,201):
    a.append(i)
  for vid, vinfo in data['database'].iteritems():
    vid_name = [v for v in video_list if vid in v]
    if len(vid_name) == 1:
      if vinfo['subset'] == split:
        # get time windows
        segment1[vid] = []
        segment2[vid] = []
        segment3[vid] = []
        for anno in vinfo['annotations']:
          start_time = anno['segment'][0]
          end_time = anno['segment'][1]
          # t = tuple()
          mlb = MultiLabelBinarizer(a)
          label = classes[anno['label']]
          # print(label)
          labels = []
          labels.append(label)
          # print(labels)
          t = tuple(labels)
          # print(t)
          b = []
          b.append(t)
          label = mlb.fit_transform(b).tolist()[0]
          print(label)
          # label = classes[anno['label']]
          # segment1[vid].append([start_time, end_time])
          segment1[vid].append(start_time)
          segment2[vid].append(end_time)
          segment3[vid].append(label)
  # sort segments by start_time
  for vid in segment1:
    # print(segment)
    # print(segment[vid])
    # segment[vid]= sorted(segment[vid], key = lambda x: float(x[0]))
    # segment[vid]= sorted(segment[vid], key=itemgetter(0))
    segment1[vid], segment2[vid], segment3[vid] = (list(t) for t in zip(*sorted(zip(segment1[vid], segment2[vid], segment3[vid]))))
    # segment[vid].sort(key=lambda x: x[0])
    
    #check the sorting
    print segment1[vid], segment2[vid], segment3[vid] 
    segment[vid] = list(zip(segment1[vid], segment2[vid]))
  return segment, segment3

def mkdir(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise

def rm(path):
  try:
    shutil.rmtree(path)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise

def ffmpeg(filename, outfile, fps):
  command = ["ffmpeg", "-i", filename, "-q:v", "1", "-r", str(fps), outfile]
  pipe = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  pipe.communicate()


def resize(filename, size = (171, 128)):
  img = cv2.imread(filename, 100)
  img2 = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
  cv2.imwrite(filename, img2, [100])

