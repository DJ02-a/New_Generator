import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2

converter = {
    # label name : original, result
    'background': {
        'label':[0, 0],
        'dilate':1,
        },
    'head'      : {
        'label':[1, 1],
        'dilate':1,
        },
    'l_brow'    : {
        'label':[2, 2],
        'dilate':1,
        },
    'r_brow'    : {
        'label':[3, 3],
        'dilate':1,
        },
    'l_eye'     : {
        'label':[4, 4],
        'dilate':5,
        },
    'r_eye'     : {
        'label':[5, 5],
        'dilate':5,
        },
    'eye_g'     : {
        'label':[6, 0],
        'dilate':1,
        },
    'l_ear'     : {
        'label':[7, 1],
        'dilate':1,
        },
    'r_ear'     : {
        'label':[8, 1],
        'dilate':1,
        },
    'ear_r'     : {
        'label':[9, 0],
        'dilate':1,
        },
    'nose'      : {
        'label':[10, 6],
        'dilate':1,
        },
    'mouth'     : {
        'label':[11, 7],
        'dilate':1,
        },
    'u_lip'     : {
        'label':[12, 7],
        'dilate':1,
        },
    'l_lip'     : {
        'label':[13, 7],
        'dilate':1,
        },
    'neck'      : {
        'label':[14, 8],
        'dilate':1,
        },
    'neck_l'    : {
        'label':[15, 0],
        'dilate':1,
        },
    'cloth'     : {
        'label':[16, 0],
        'dilate':1,
        },
    'hair'      : {
        'label':[17, 9],
        'dilate':1,
        },
    'hat'       : {
        'label':[18, 0],
        'dilate':1,
        }, 
}



def label_converter(mask):
    _mask = np.array(mask)
    canvas = np.zeros_like(_mask)
    for part_name in converter:
        before_label, after_label = converter[part_name]['label']
        canvas = np.where(_mask==before_label, after_label, canvas)
        
    return canvas

def face_mask2one_hot(label, size):
    label = torch.tensor(label, dtype=torch.int64)
    c = 11
    h,w = label.size()
    label = torch.reshape(label,(1,1,h,w))
    
    one_hot = torch.FloatTensor(1, c, h, w).zero_()
    _one_hot = one_hot.scatter_(1, label, 1.0)
    _one_hot = F.interpolate(_one_hot,(size,size),mode='nearest')
    return _one_hot.squeeze()
    
    
# input numpy (h w)shape
def part_mask2one_hot(part_mask, part=None, dilate=False):
    assert part in converter.keys()
    part_name = part
    part_mask = np.array(part_mask)
    
    # dilate
    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        iter = converter[part_name]['dilate']
        part_mask = cv2.dilate(part_mask.astype(np.uint8), kernel, iterations=iter)


    # to one hot
    h, w = part_mask.shape
    c = 11
    label = converter[part_name]['label'][1]
    
    canvas = torch.zeros((c,h,w))
    part_mask = torch.tensor(part_mask/255, dtype=torch.int32)
    
    canvas[1] = (1 - part_mask)
    canvas[label] = part_mask
    return canvas        
