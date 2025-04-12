# ==============================================================
#   Copyright (C) 2025 All rights reserved.
#
#   name：  learn_transformer_pe_sin.py
#   author：hongleifu
#   date：  2025/4/6
#   describe：learn transformer's position encode
# ================================================================

import os
import numpy as np
import math
import display

def create_pos_encode_element_sincos(token_pos=1,token_dim=100,token_dim_pos=0):
    """
    create one element in pos encode of one token.
    :param:
             token_pos: token pos in whole sentence
             token_dim: dim of pos encode of each token
             token_dim_pos: index in one pos encode
    :return: value of one element in pos encode
             value = sin(token_pos/pow(10000,(i/token_dim))) if token_dim_pos==2i
             value = cos(token_pos/pow(10000,(i/token_dim))) if token_dim_pos==2i+1
    """
    flag = 'sin'
    if token_dim_pos%2 == 1:
        flag = 'cos'
    i = token_dim_pos/2
    if flag == 'sin':
        v = math.sin(float(token_pos) / pow(10000, float(i) / token_dim))
    else:
        v = math.cos(float(token_pos) / pow(10000, float(i) / token_dim))
    return v

def create_pos_encode_sincos(token_pos=1,token_dim=100):
    """
    create pos encode of one token.
    :param:
             token_pos: token pos in whole sentence
             token_dim: dim of pos encode of each token
    :return: value of pos encode of one token
    """
    encodes = np.zeros(token_dim)
    for i in range(0,token_dim):
        encodes[i] = create_pos_encode_element_sincos(token_pos=token_pos,token_dim=token_dim,token_dim_pos=i)
    return encodes

def display_pos_encode_sincos(token_pos=1,token_dim=100,show_line=False):
    """
    :param:
             token_pos: token pos in whole sentence
             token_dim: dim of pos encode of each token
    :return: none
    """
    encodes = create_pos_encode_sincos(token_pos=token_pos,token_dim=token_dim)
    xs = []
    ys = []
    for i in range(0,token_dim):
        xs.append(i)
        ys.append(encodes[i])
    if show_line:
        display.show_pts_line(xs,ys)
    else:
        display.show_pts(xs,ys)
    return None

def display_poses_encode_sincos(token_poses=[1,2,3,4,5],token_dim=100):
    """
    :param:
             token_poses: token poses in whole sentence
             token_dim: dim of pos encode of each token
    :return: none
    """
    x_group = []
    y_group = []
    for pos in token_poses:
        encodes = create_pos_encode_sincos(token_pos=pos,token_dim=token_dim)
        xs = []
        ys = []
        for i in range(0,token_dim):
            xs.append(i)
            ys.append(encodes[i])
        x_group.append(xs)
        y_group.append(ys)
    display.show_group_pts(x_group=x_group,y_group=y_group)
    return None

def display_pos_sub_sincos(poses=[1,2],token_dim=100):
    """
    display two poses encode sub
    :param:
             poses: two token poses in whole sentence to caculate there's sub
             token_dim: dim of pos encode of each token
    :return: none
    """
    encode1 = create_pos_encode_sincos(token_pos=poses[0],token_dim=token_dim)
    encode2 = create_pos_encode_sincos(token_pos=poses[1],token_dim=token_dim)
    encode_sub = encode1-encode2
    xs = []
    ys = []
    for i in range(0, token_dim):
        xs.append(i)
        ys.append(encode_sub[i])
    display.show_pts(xs, ys)

def display_poses_sub_sincos(poses=[[1,2],[10,11]],token_dim=100):
    """
    display many poses encode sub
    :param:
             poses: list of two token poses
             token_dim: dim of pos encode of each token
    :return: none
    """
    x_group = []
    y_group = []
    for sub_poses in poses:
        encode1 = create_pos_encode_sincos(token_pos=sub_poses[0], token_dim=token_dim)
        encode2 = create_pos_encode_sincos(token_pos=sub_poses[1], token_dim=token_dim)
        encode_sub = encode1 - encode2
        xs = []
        ys = []
        for i in range(0,token_dim):
            xs.append(i)
            ys.append(encode_sub[i])
        x_group.append(xs)
        y_group.append(ys)
    display.show_group_pts(x_group=x_group,y_group=y_group)

if __name__ == "__main__":
    display_pos_encode_sincos(token_pos=1, token_dim=100,show_line=False)
    # display_poses_encode_sincos(token_poses=[11,12],token_dim=100)
    # display_pos_sub_sincos(poses=[1, 2], token_dim=100)
    # display_poses_sub_sincos(poses=[[1,2],[11,12]],token_dim=100)



