# coding: utf-8
import torch
import numpy as np
import math

def point_form(boxes):
    """
    Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    :param boxes: (tensor) center-size default boxes from priorbox layers.
    :return: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:]/2), 1)   # xmax, ymax

def center_size(boxes):
    """
    Convert prior_boxes to (cx, cy, w, h)
    representation for comparion to center-size form ground truth data.
    :param boxes: (tensor) point_form boxes
    :return: (tensor) Converted xmin, ymin, xmax, ymax form of boxes
    """
    return torch.cat([(boxes[:, :2] + boxes[:, 2:])/2,  # cx, cy
                      boxes[:, :2] - boxes[:, 2:]], 1)   # w, h

def intersect(box_a, box_b):
    """
    We resize both tensors to [A, B, 2] without new malloc:
    [A, 2] -> [A, 1, 2] -> [A, B, 2]
    [B, 2] -> [1, B, 2] -> [A, B, 2]
    Then we compute the area of intersect between box_a and box_b
    :param box_a: (tensor) bounding boxes, Shape: [A, 4]
    :param box_b: (tensor) bounding boxes, Shape: [B, 4]
    :return: (tensor) intersection area, Shape: [A, B]
              inter[i, j] = intersection area between i_th box and j_th box, where i in A, j in B
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """
    Compute the jaccard overlap of twe sets of boxes. The jaccard overlap
    is simply the intersection over union of twe boxes. Here we operate on
    ground truth boxes and default boxes.
    e.g.:
      A qie B / A bing B = A qie B / (area(A) + area(B) - A qie B)
    :param box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects, 4]
    :param box_b: (tensor) Prior boxes from priorbox layer, shape: [num_priors, 4]
    :return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, ]))