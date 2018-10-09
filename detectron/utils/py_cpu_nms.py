# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from IPython import embed
from detectron.core.config import cfg

def soft(dets, confidence=None, ax = None):
    thresh = cfg.STD_TH #.6
    score_thresh = .7
    sigma = .5 
    N = len(dets)
    x1 = dets[:, 0].copy()
    y1 = dets[:, 1].copy()
    x2 = dets[:, 2].copy()
    y2 = dets[:, 3].copy()
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ious = np.zeros((N,N))
    for i in range(N):
        xx1 = np.maximum(x1[i], x1)
        yy1 = np.maximum(y1[i], y1)
        xx2 = np.minimum(x2[i], x2)
        yy2 = np.minimum(y2[i], y2)

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas - inter)
        ious[i,:] = ovr
             
    i = 0
    while i < N:
        maxpos = dets[i:N, 4].argmax()
        maxpos += i
        dets[[maxpos,i]] = dets[[i,maxpos]]
        areas[[maxpos,i]] = areas[[i,maxpos]]
        confidence[[maxpos,i]] = confidence[[i,maxpos]]
        ious[[maxpos,i]] = ious[[i,maxpos]]
        ious[:,[maxpos,i]] = ious[:,[i,maxpos]]

        ovr_bbox = np.where((ious[i, :N] > thresh))[0]
        avg_std_bbox = (dets[ovr_bbox, :4] / confidence[ovr_bbox]).sum(0) / (1/confidence[ovr_bbox]).sum(0)
        if cfg.STD_NMS:
            dets[i,:4] = avg_std_bbox
        else:
            assert(False)

        areai = areas[i]
        pos = i + 1
        while pos < N:
            if ious[i , pos] > 0:
                ovr =  ious[i , pos]
                dets[pos, 4] *= np.exp(-(ovr * ovr)/sigma)
                if dets[pos, 4] < 0.001:
                    dets[[pos, N-1]] = dets[[N-1, pos]]
                    areas[[pos, N-1]] = areas[[N-1, pos]]
                    confidence[[pos, N-1]] = confidence[[N-1, pos]]
                    ious[[pos, N-1]] = ious[[N-1, pos]]
                    ious[:,[pos, N-1]] = ious[:,[N-1, pos]]
                    N -= 1
                    pos -= 1
            pos += 1
        i += 1
    keep=[i for i in range(N)]
    return dets[keep], keep

def greedy(dets, thresh=None, confidence=None, ax = None):
    """ greedy QUBO solver for NMS"""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    if len(scores) == 0:
        return []
    elif len(scores) == 1:
        return [0]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ious = np.zeros([len(x1), len(x1)])
    w2 = .42
    for i in range(len(x1)):
        xx1 = np.maximum(x1[i], x1)
        yy1 = np.maximum(y1[i], y1)
        xx2 = np.minimum(x2[i], x2)
        yy2 = np.minimum(y2[i], y2)

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas - inter)
        for j in range(len(ovr)):
            if i == j:
                ious[i, i] = -(1-w2) * scores[i]
                ious[i,j] += w2 * ovr[j]
            else:
                ious[i,j] = w2 * ovr[j]
    score = 0
    keep = []
    remain = list(range(len(x2)))
    while len(remain) > 0:
        if len(keep) == 0:
            inc = np.diag(ious)[remain]
        else:
            try:
                d = ious[remain][:, keep]
                if len(d.shape) == 2:
                    d = d.sum(1)
                inc = d+np.diag(ious)[remain]
            except:
                embed()
        if inc.min() >= 0:
            break
        idx = inc.argmin()
        obj = remain[idx]
        remain.remove(obj)
        keep.append(obj)
    if len(keep) !=0:
        print(len(keep), len(x1))

    return keep

def qb(dets, thresh=None, confidence=None, ax = None):
    """ QUBO solver for NMS"""
    from dwave_qbsolv import QBSolv
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    if len(scores) == 0:
        return []
    elif len(scores) == 1:
        return [0]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ious = {}#np.zeros([len(x1), len(x1)])
    w2 = 0.4
    for i in range(len(x1)):
        xx1 = np.maximum(x1[i], x1)
        yy1 = np.maximum(y1[i], y1)
        xx2 = np.minimum(x2[i], x2)
        yy2 = np.minimum(y2[i], y2)

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas - inter)
        for j in range(len(ovr)):
            if i == j:
                ious[i, i] = -(1-w2) * scores[i]
                ious[i,j] += w2 * ovr[j]
            else:
                ious[i,j] = w2 * ovr[j]
    response = QBSolv().sample_qubo(ious, verbosity=0)
    samples = list(response.samples())[0]
    keep = []
    for i in range(len(x1)):
        if samples[i] == 1:
            keep.append(i)
    return keep
