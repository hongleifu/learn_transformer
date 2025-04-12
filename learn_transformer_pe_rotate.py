# ==============================================================
#   Copyright (C) 2025 All rights reserved.
#
#   name：  learn_transformer_pe_rotate.py
#   author：hongleifu
#   date：  2025年4月1日
#   describe：learn transformer rotate position encode
# ================================================================


import os
import numpy as np
import math
import display

def create_cos_ratio(token_pos=1,token_dim=100):
    """
    create pos encode cos ratio of one token.
    :param:
             token_pos: token pos in whole sentence
             token_dim: dim of pos encode of each token
    :return: value of cos ratio
    """
    encodes = np.zeros(token_dim)
    for i in range(0,token_dim):
        encodes[i] = math.cos(float(token_pos) / pow(10000, float(i/2) / token_dim))
    return encodes

def create_sin_ratio(token_pos=1,token_dim=100):
    """
    create pos encode sin ratio of one token.
    :param:
             token_pos: token pos in whole sentence
             token_dim: dim of pos encode of each token
    :return: value of sin ratio
    """
    encodes = np.zeros(token_dim)
    for i in range(0,token_dim):
        encodes[i] = math.sin(float(token_pos) / pow(10000, float(i/2) / token_dim))
    return encodes

def create_fusion_embedding(token_embedding,token_pos,token_dim=100):
    """
    create fusion_embedding of token embbeding and pos embbdeing.
    :param:
             token_embedding: token pos in whole sentence
             word_pos:1 2 3...
             token_dim: dim of pos encode of each token
    :return: value of fusion_embedding
    """
    cos_ratio = create_cos_ratio(token_pos=token_pos,token_dim=token_dim)
    sin_ratio = create_sin_ratio(token_pos=token_pos, token_dim=token_dim)

    sin_coef = np.zeros(token_dim)
    for i in range(0,token_dim):
        if i % 2 == 0:
            if i+1 < token_dim:
                sin_coef[i] = token_embedding[i+1]*(-1.0)
            else:
                sin_coef[i] = token_embedding[i+1]
        else:
            sin_coef[i] = token_embedding[i-1]
    cos_coef = token_embedding

    fusion = cos_coef * cos_ratio + sin_coef * sin_ratio
    return fusion

def display_pe_rotate(token_pos=1,token_dim=100,show_line=False):
    """
    :param:
             token_pos: token pos in whole sentence
             token_dim: dim of pos encode of each token
    :return: none
    """
    token_ebd = np.ones(token_dim)
    fusion = create_fusion_embedding(token_embedding=token_ebd,token_pos=token_pos,token_dim=token_dim)
    xs = []
    ys = []
    for i in range(0,token_dim):
        xs.append(i)
        ys.append(fusion[i])
    if show_line:
        display.show_pts_line(xs,ys)
    else:
        display.show_pts(xs,ys)
    return None

def display_pes_rotate(token_poses=[1,2,3,4,5],token_dim=100):
    """
    :param:
             token_poses: token poses in whole sentence
             token_dim: dim of pos encode of each token
    :return: none
    """
    x_group = []
    y_group = []
    for pos in token_poses:
        token_ebd = np.ones(token_dim)
        encodes = create_fusion_embedding(token_embedding=token_ebd, token_pos=pos, token_dim=token_dim)
        xs = []
        ys = []
        for i in range(0,token_dim):
            xs.append(i)
            ys.append(encodes[i])
        x_group.append(xs)
        y_group.append(ys)
    display.show_group_pts(x_group=x_group,y_group=y_group)
    return None

def display_pes_dot(poses=[[1,5],[11,15]],token_dim=100):
    """
    display list of two poses encode dot, and each group two poes sub is same.
    :param:
             poses: two token poses in whole sentence to caculate there's dot.
             token_dim: dim of pos encode of each token
    :return: none
    """
    xs = []
    ys = []
    token_ebd = np.ones(token_dim)
    for i in range(len(poses)):
        item = poses[i]
        pe0 = create_fusion_embedding(token_embedding=token_ebd, token_pos=item[0], token_dim=token_dim)
        pe1 = create_fusion_embedding(token_embedding=token_ebd, token_pos=item[1], token_dim=token_dim)

        xs.append(i+1)
        ys.append(np.dot(pe0,pe1))
    display.show_pts(xs, ys)

if __name__ == "__main__":
    # for i in range(2,3):
    #     display_pe_rotate(token_pos=i,token_dim=100,show_line=True)
    # display_pes_rotate(token_poses=[1,2,3,4,5],token_dim=100)

    poes = [[1, 2], [3, 4],[5, 10], [11, 16],[20,30], [40, 50],[40,60],[50,70]]
    display_pes_dot(poses=poes, token_dim=100)



