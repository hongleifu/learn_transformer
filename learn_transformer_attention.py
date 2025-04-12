# ==============================================================
#   Copyright (C) 2025 All rights reserved.
#
#   name：  learn_transformer_attention.py
#   author：hongleifu
#   date：  2025年12月30日
#   describe：learn transformer attention.
#
#
# ================================================================


import os
import numpy as np
import math
import display

def create_randn_tokens_embedding(seq_len = 10,token_dim = 1):
    """
    direct create random tokens embedding sequnce.
    and value of each embedding's postion match gaussion distribution.
    :param:
        seq_len: length of token's sequence
        token_dim: dim of each token embedding
    :return: created tokens_embedding
    """
    tokens_embedding = np.random.randn(seq_len, token_dim)
    return tokens_embedding

def q_element_dot_k(q,k,q_pos):
    """
    cal dot of one element in q and k
    :param:
        q: query embedding sequence
        k: key embedding sequence
        q_pos:index of element to be doted in query embedding sequence
    :return: dot result
    """
    dis = np.dot(k,q[q_pos])
    return dis

def display_q_element_dot_k(seq_len = 10,token_dim = 1,q_pos = 0,scale='sqrt_dim'):
    """
    display info of q_element dot k. here q is same to k.
    create randn tokens embedding first, then cal dot of one element of q and k.
    :param:
        seq_len: length of token's sequence
        token_dim: dim of each token embedding
        q_pos: index of element to be doted in query embedding sequence
        scale: scale of q*k.
               'sqrt_dim' = sqrt(token_dim)
               'none' = 1
               'dim' = token_dim
    :return: None
    """
    #create tokens encoding
    tokens_embedding = create_randn_tokens_embedding(seq_len=seq_len,token_dim=token_dim)
    dis = q_element_dot_k(q = tokens_embedding,k=tokens_embedding,q_pos=q_pos)

    if scale == 'sqrt_dim':
        dis = dis/np.sqrt(token_dim)
    elif scale == 'dim':
        dis = dis / token_dim
    elif scale == 'none':
        dis = dis

    dis = np.around(dis, decimals=2)

    nums,counts = np.unique(dis, return_counts=True)
    counts = counts/float(seq_len)
    display.show_pts(nums,counts)
    print('std error is:',np.std(dis))
    print('square error is:', np.var(dis))
    return None


if __name__ == "__main__":
    # scale = 'none'
    # scale = 'dim'
    scale = 'sqrt_dim'
    display_q_element_dot_k(seq_len=5000,token_dim=100,q_pos=2,scale=scale)


