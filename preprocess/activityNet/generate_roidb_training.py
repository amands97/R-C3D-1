# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
import copy
import json
import cPickle
import subprocess
import numpy as np
# import cv2
from util import *

FPS = 25
LENGTH = 100
min_length = 0
overlap_thresh = 0.7
STEP = LENGTH / 4
WINS = [LENGTH * 8]
print(WINS)
META_FILE = './activity_net.v1-3.min.json'
data = json.load(open(META_FILE))

print ('Generate Classes')
classes = generate_classes(data)

print ('Generate Training Segments')
train_segment = generate_segment('training', data, classes)

path = './preprocess/activityNet/frames/'
def generate_roi(rois1, rois2, video, start, end, stride, split):
  tmp = {}
  tmp['wins'] = ( rois1 - start ) / stride
  tmp['durations'] = tmp['wins'][:,1] - tmp['wins'][:,0]
  tmp['gt_classes'] = rois2
  tmp['max_classes'] = rois2
  tmp['max_overlaps'] = np.ones(len(rois1))
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, start, end, stride]])
  tmp['bg_name'] = path + split + '/' + video
  tmp['fg_name'] = path + split + '/' + video
  if not os.path.isfile('../../' + tmp['bg_name'] + '/image_' + str(end-1).zfill(5) + '.jpg'):
    print ('../../' + tmp['bg_name'] + '/image_' + str(end-1).zfill(5) + '.jpg')
    raise
  return tmp

def generate_roidb(split, segment1, segment2):
  VIDEO_PATH = 'frames/%s/' % split
  video_list = set(os.listdir(VIDEO_PATH))
  duration = []
  roidb = []
  for vid in segment1:
    if vid in video_list:
      length = len(os.listdir('./frames/' + split + '/' + vid))
      db1 = np.array(segment1[vid])
      db2 = np.array(segment2[vid])
      if len(db1) == 0:
        print("0 length")
        continue
      db1[:,:2] = db1[:,:2] * FPS

      for win in WINS:
        stride = win / LENGTH
        step = stride * STEP
        # Forward Direction
        for start in xrange(0, max(1, length - win + 1), step):
          end = min(start + win, length)
          assert end <= length
          truelist = (np.logical_not(np.logical_or(db1[:,0] >= end, db1[:,1] <= start)))
          rois1 = db1[truelist]
          rois2 = db2[truelist]
          print rois1
          print rois2

          # print("db", db1)
          # Remove duration less than min_length
          if len(rois1) > 0:
            duration = rois1[:,1] - rois1[:,0]
            rois1 = rois1[duration >= min_length]
            rois2 = rois2[duration >= min_length]

          # Remove overlap less than overlap_thresh
          if len(rois1) > 0:
            time_in_wins = (np.minimum(end, rois1[:,1]) - np.maximum(start, rois1[:,0]))*1.0
            overlap = time_in_wins / (rois1[:,1] - rois1[:,0])
            assert min(overlap) >= 0
            assert max(overlap) <= 1
            rois1 = rois1[overlap >= overlap_thresh]
            rois2 = rois2[overlap >= overlap_thresh]

          # Append data
          if len(rois1) > 0:
            rois1[:,0] = np.maximum(start, rois1[:,0])
            rois1[:,1] = np.minimum(end, rois1[:,1])
            tmp = generate_roi(rois1, rois2, vid, start, end, stride, split)
            roidb.append(tmp)
            if USE_FLIPPED:
               flipped_tmp = copy.deepcopy(tmp)
               flipped_tmp['flipped'] = True
               roidb.append(flipped_tmp)

        # Backward Direction
        for end in xrange(length, win-1, - step):
          start = end - win
          assert start >= 0
          rois1 = db1[np.logical_not(np.logical_or(db1[:,0] >= end, db1[:,1] <= start))]
          rois2 = db2[np.logical_not(np.logical_or(db1[:,0] >= end, db1[:,1] <= start))]

          # Remove duration less than min_length
          if len(rois1) > 0:
            duration = rois1[:,1] - rois1[:,0]
            rois1 = rois1[duration > min_length]

          # Remove overlap less than overlap_thresh
          if len(rois1) > 0:
            time_in_wins = (np.minimum(end, rois1[:,1]) - np.maximum(start, rois1[:,0]))*1.0
            overlap = time_in_wins / (rois1[:,1] - rois1[:,0])
            assert min(overlap) >= 0
            assert max(overlap) <= 1
            rois1 = rois1[overlap > overlap_thresh]
            rois2 = rois2[overlap > overlap_thresh]

          # Append data
          if len(rois1) > 0:
            rois1[:,0] = np.maximum(start, rois1[:,0])
            rois1[:,1] = np.minimum(end, rois1[:,1])
            tmp = generate_roi(rois1, rois2, vid, start, end, stride, split)
            roidb.append(tmp)
            if USE_FLIPPED:
               flipped_tmp = copy.deepcopy(tmp)
               flipped_tmp['flipped'] = True
               roidb.append(flipped_tmp)

  return roidb

USE_FLIPPED = True      
train_roidb = generate_roidb('training', train_segment[0], train_segment[1])

print ("Save dictionary")
cPickle.dump(train_roidb, open('train_data_3fps_flipped.pkl','w'), cPickle.HIGHEST_PROTOCOL)


# def generate_roidb(split, segment1, segment2):
#   VIDEO_PATH = 'frames/%s/' % split
#   video_list = set(os.listdir(VIDEO_PATH))
#   duration = []
#   roidb = []
#   for vid in segment1:
#     if vid in video_list:
#       length = len(os.listdir('./frames/' + split + '/' + vid))
#       # db = np.array(segment[vid])
#       # print("\n\n\n\n\n")      
#       # print segment[vid]
#       # print("\n\n\n\n\n")
#       # for xi in segment[vid]:
#       #   print("\n\n")
#       #   print "xi",xi
#       #   print("\n\n")
        
#         # print(np.array(xi))
#       # print([np.array(xi) for xi in segment[vid]])
#       # db=np.array([np.array(xi) for xi in segment[vid]])
#       # for i, xi in enumerate(segment[vid]):
#       #     for j, x2i in enumerate(xi):
#       #       if x2i.__class__.__name__ in ('list'):
#       #         continue
#       #       segment[vid][i][j] = [segment[vid][i][j]]
#       # db=np.array([np.array(xi) for xi in segment[vid]])
#       # print(segment[vid])
#       # db=np.array([np.array([np.array(x2i) for xi2 in xi]) for xi in segment[vid]])
#       # print(segment[vid])
#       db1 = segment1[vid]
#       # db=np.array([np.array(xi) for xi in segment[vid][0]])
#       # db = np.array(segment[vid])
#       # print(db[0].shape)
#       # print(db[0][2])
#       # print(db[:,:2])
#       if len(db) == 0:
#         continue
#       # print(db)
      
#       # print(db)
#       for row in db:
#         row[0] = row[0]*FPS
#         row[1] = row[1]*FPS
#       # db[:,:2] = db[:,:2] * FPS
#       # print(db)
#       for win in WINS:
#         stride = win / LENGTH
#         step = stride * STEP
#         # Forward Direction
#         for start in xrange(0, max(1, length - win + 1), step):
#           end = min(start + win, length)
#           assert end <= length
#           # rois = db[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] <= start))]
#           rois = []
#           # be careful of the less than and equal to here
#           for row in db:
#             if row[0]< end and row[1] > start:
#               rois.append(row)
#           # print(rois)
#           # Remove duration less than min_length
#           # if len(rois) > 0:
#           #   duration = rois[:,1] - rois[:,0]
#           #   rois = rois[duration >= min_length]

#           # Remove overlap less than overlap_thresh
#           # print(rois[:,1])
#           print(rois[0][1])
#           if len(rois) > 0:
#             time_in_wins = (np.minimum(end, rois[:,1]) - np.maximum(start, rois[:,0]))*1.0
#             overlap = time_in_wins / (rois[:,1] - rois[:,0])
#             assert min(overlap) >= 0
#             assert max(overlap) <= 1
#             rois = rois[overlap >= overlap_thresh]

#           # Append data
#           if len(rois) > 0:
#             rois[:,0] = np.maximum(start, rois[:,0])
#             rois[:,1] = np.minimum(end, rois[:,1])
#             tmp = generate_roi(rois, vid, start, end, stride, split)
#             roidb.append(tmp)
#             if USE_FLIPPED:
#                flipped_tmp = copy.deepcopy(tmp)
#                flipped_tmp['flipped'] = True
#                roidb.append(flipped_tmp)

#         # Backward Direction
#         # for end in xrange(length, win-1, - step):
#         #   start = end - win
#         #   assert start >= 0
#         #   rois = db[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] <= start))]

#         #   # Remove duration less than min_length
#         #   # if len(rois) > 0:
#         #   #   duration = rois[:,1] - rois[:,0]
#         #   #   rois = rois[duration > min_length]

#         #   # Remove overlap less than overlap_thresh
#         #   if len(rois) > 0:
#         #     time_in_wins = (np.minimum(end, rois[:,1]) - np.maximum(start, rois[:,0]))*1.0
#         #     overlap = time_in_wins / (rois[:,1] - rois[:,0])
#         #     assert min(overlap) >= 0
#         #     assert max(overlap) <= 1
#         #     rois = rois[overlap > overlap_thresh]

#         #   # Append data
#         #   if len(rois) > 0:
#         #     rois[:,0] = np.maximum(start, rois[:,0])
#         #     rois[:,1] = np.minimum(end, rois[:,1])
#         #     tmp = generate_roi(rois, vid, start, end, stride, split)
#         #     print(tmp)
#         #     roidb.append(tmp)
#         #     if USE_FLIPPED:
#         #        flipped_tmp = copy.deepcopy(tmp)
#         #        flipped_tmp['flipped'] = True
#         #        roidb.append(flipped_tmp)

#   return roidb

