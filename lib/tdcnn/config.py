# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

"""R-C3D config system.

This file specifies default config options for R-C3D. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn.config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()

# Length of frame to use during training (can list multiple lengths)
__C.TRAIN.LENGTH = (512,)

# Size of input frames and crop size
__C.TRAIN.FRAME_SIZE = (128, 171)
__C.TRAIN.CROP_SIZE  = 112

# Images to use per minibatch
__C.TRAIN.VIDEO_BATCH = 1

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128


# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train time windows regressors
__C.TRAIN.TWIN_REG = True

# Overlap required between a ROI and ground-truth window in order for that
# ROI to be used as a time window regression training example
__C.TRAIN.TWIN_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 1000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
__C.TRAIN.USE_PREFETCH = False

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
# __C.TRAIN.TWIN_NORMALIZE_TARGETS = True
__C.TRAIN.TWIN_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
__C.TRAIN.TWIN_INSIDE_WEIGHTS = (1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (TWIN_NORMALIZE_TARGETS must also be True)
__C.TRAIN.TWIN_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.TWIN_NORMALIZE_MEANS = (0.0, 0.0)
__C.TRAIN.TWIN_NORMALIZE_STDS = (0.1, 0.2)

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'selective_search'

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 8
## Deprecated (outside weights)
__C.TRAIN.RPN_TWIN_INSIDE_WEIGHTS = (1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

__C.TRAIN.CINPUT = False
__C.TRAIN.RANDOM = False
__C.TRAIN.OHEM_USE_NMS = True
__C.TRAIN.OHEM_NMS_THRESH = 0.7
__C.TRAIN.OHEM_LOSS_ONLY_CLASSIFICATION = False
__C.TRAIN.OHEM_LOSS_ONLY_LOCALIZATION = False



#
# Testing options
#

__C.TEST = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.LENGTH = (512,)

# Size of input frames and crop size
__C.TEST.FRAME_SIZE = (128, 171)
__C.TEST.CROP_SIZE  = 112

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.TWIN_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'selective_search'

# NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 8


#
# MISC
#

__C.NUM_CLASSES = 21

__C.INPUT = 'frames'

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/8 is correct for C3D
__C.DEDUP_BOXES = 1./8.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[90, 98, 102]]])
__C.PIXEL_MEANS_FLOW = np.array([128])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Frame per second
__C.FPS = 25

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
