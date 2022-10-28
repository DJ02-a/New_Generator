import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# Face Parsing model to new label
face_parsing_converter = {
    0 : 0,
    1 : 1,
    2 : 2,
    3 : 3,
    4 : 4,
    5 : 5,
    6 : 0,
    7 : 1,
    8 : 1,
    9 : 0,
    10 : 9,
    11 : 6,
    12 : 7,
    13 : 8,
    14 : 11,
    15 : 0,
    16 : 0,
    17 : 10,
    18 : 0
}

# Celeba HQ to new label
celeba_hq_converter = {
    0 : 0,
    1 : 1,
    2 : 1,
    3 : 0,
    4 : 4,
    5 : 5,
    6 : 2,
    7 : 3,
    8 : 1,
    9 : 1,
    10 : 6,
    11 : 7,
    12 : 8,
    13 : 9,
    14 : 0,
    15 : 0,
    16 : 0,
    17 : 1,
    18 : 0
}

def label_converter(before_label):
    _before_label = np.array(before_label)
    canvas = np.zeros_like(_before_label)
    for idx in face_parsing_converter:
        canvas = np.where(_before_label==idx, face_parsing_converter[idx], canvas)
        
    # 4 5
    kernel = np.ones((3, 3), np.uint8)
    for dilate_idx in [4,5]:
        eye_mask = np.where(canvas==dilate_idx,1,0)
        dilated_eye_mask = cv2.dilate(eye_mask.astype(np.uint8), kernel, iterations=3)
        canvas = canvas * (1-dilated_eye_mask) + (dilated_eye_mask * dilate_idx) * (dilated_eye_mask)

    return canvas

def to_one_hot(Xt):
    Xt_ = torch.tensor(Xt, dtype=torch.int64)
    c = max(face_parsing_converter.values())+1
    h,w = Xt_.size()
    Xt_ = torch.reshape(Xt_,(1,1,h,w))
    one_hot_Xt = torch.FloatTensor(1, c, h, w).zero_()
    one_hot_Xt_ = one_hot_Xt.scatter_(1, Xt_, 1.0)
    one_hot_Xt_ = F.interpolate(one_hot_Xt_,(h,w),mode='bilinear')
    return one_hot_Xt_.squeeze() 

def vis_mask(one_hot_mask, name=None, grid_result=False):
    os.makedirs(f'./result/{name}', exist_ok=True)
    c, _, _ = one_hot_mask.shape

    grid = []
    for idx in range(c):
        one_ch = one_hot_mask[idx]
        _one_ch = np.array(one_ch)
        grid.append(_one_ch * 255)
        # cv2.imwrite(f'./result/{name}/{idx}.png', _one_ch * 255)
    if grid_result: cv2.imwrite(f'./result/{name}/grid.png', np.concatenate(grid,axis=-1))