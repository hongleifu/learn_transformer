# ==============================================================
#   Copyright (C) 2025 All rights reserved.
#
#   name：  learn_transformer_pe_relative.py
#   author：hongleifu
#   date：  2025年4月1日
#   describe：learn relative positon encode of transformer
# ================================================================


import os
import numpy as np
import math
import display

def create_pos_encode(token_dim=100,relative_pos=1):
    """
    create pos encode of relative pos.
    :param:
             token_dim: dim of pos encode of each token
             relative_pos: relative positon of two token
    :return: value of pos encode
    """
    encode = np.zeros(token_dim)
    for i in range(0,token_dim):
        encode[i] = math.sin(float(i)/token_dim)
    if relative_pos == 0:
        relative_pos = 1
    encode = encode/float(relative_pos)
    return encode

def display_pos_encode(token_dim=100,relative_pos=1,show_line=False):
    """
    :param:
             relative_pos: relative positon of two token
             token_dim: dim of pos encode of each token
    :return: none
    """
    encodes = create_pos_encode(token_dim=token_dim,relative_pos=relative_pos)
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

def display_poses_encode(token_dim=100,relative_poses=[1,2,3,4,5]):
    """
    :param:
             relative_poses: different relative pose of token
             token_dim: dim of pos encode of each token
    :return: none
    """
    x_group = []
    y_group = []
    for pos in relative_poses:
        encodes = create_pos_encode(token_dim=token_dim,relative_pos=pos)
        xs = []
        ys = []
        for i in range(0,token_dim):
            xs.append(i)
            ys.append(encodes[i])
        x_group.append(xs)
        y_group.append(ys)
    display.show_group_pts(x_group=x_group,y_group=y_group)
    return None

if __name__ == "__main__":
    # display_pos_encode(token_dim=100, relative_pos=1, show_line=False)
    display_poses_encode(token_dim=100, relative_poses=[1, 2, 3, 4, 5])



