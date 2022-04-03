# Config file

import tensorflow as tf
from util.feature_columns import SparseFeat, VarLenSparseFeat
import os


class Config:
    ########################################################################
    # GPU
    ########################################################################
    device_id = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    ########################################################################
    # Data config
    ########################################################################
    build_data = {
        'i_st': tf.constant([[85, 159, 257, 1299, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [741, 97, 493, 1153, 534, 1430, 275, 630, 0, 0, 0, 0, 0, 0, 0],
                             [191, 615, 590, 737, 493, 357, 715, 1739, 1924, 49, 275, 468, 0, 0, 0],
                             [49, 617, 376, 733, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [497, 85, 357, 278, 250, 2170, 294, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'sb': tf.constant([[109797743],
                           [171452880],
                           [71761253],
                           [100650287],
                           [236091878]]),
        'u_ac': tf.constant(
            [[47287, 61186, 65553, 50497, 31836, 55670, 44154, 22597, 57749, 55581, 59489, 21752, 19664, 35592, 0, 0, 0,
              0,
              0, 0, 0, 0, 0, 0, 0],
             [68968, 48294, 50504, 28826, 38376, 25267, 60890, 37718, 46099, 59738, 39461, 39836, 43681,
              55922, 22462, 37082, 32379, 39927, 0, 0, 0, 0, 0, 0, 0],
             [37271, 47318, 25009, 31910, 64472, 39452, 68968, 48294, 50504, 28826, 38376, 25267, 22767,
              20966, 19459, 20922, 29415, 45332, 60890, 37718, 46099, 59738, 39461, 39836, 41541],
             [72647, 73398, 62262, 62866, 67132, 24004, 45621, 38138, 69638, 28509, 28526, 61602, 38708,
              44262, 61705, 50533, 50309, 25354, 23312, 70089, 72104, 50683, 32759, 50926, 38448],
             [20922, 32782, 19806, 51709, 50246, 47306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0]]),
        'u_ci': tf.constant([[4414],
                             [1219],
                             [1102],
                             [1],
                             [1119]]),
        'u_co': tf.constant([[2, 0, 0],
                             [2, 3, 0],
                             [2, 0, 0],
                             [2, 3, 0],
                             [2, 0, 0]]),
        'u_di': tf.constant([[15870, 8735, 10770, 16683, 0],
                             [16135, 2399, 6709, 15473, 0],
                             [4635, 16135, 1994, 2399, 11231],
                             [14901, 7466, 13558, 11043, 4684],
                             [10102, 0, 0, 0, 0]]),
        'u_mt': tf.constant([[13, 36, 22, 11, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [13, 36, 24, 44, 22, 233, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [22, 36, 11, 13, 42, 84, 24, 44, 23, 87, 98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [207, 44, 13, 16, 24, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [24, 22, 45, 13, 260, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'u_pr': tf.constant([[1033],
                             [1021],
                             [1012],
                             [1009],
                             [1013]]),
        'u_st': tf.constant(
            [[85, 159, 257, 1299, 585, 278, 1125, 615, 737, 716, 393, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0],
             [1237, 85, 257, 294, 778, 427, 1179, 376, 278, 993, 415, 2874, 282, 1153, 534, 664, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [122, 572, 29, 85, 1466, 585, 278, 250, 49, 257, 439, 1237, 294, 778, 2675, 115, 357, 324, 437,
              597, 427, 1179, 376, 993, 191, 615, 590, 737, 493, 715],
             [275, 617, 493, 24, 376, 733, 737, 715, 597, 427, 357, 415, 351, 3852, 139, 603, 85, 1554, 49,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [497, 85, 357, 278, 250, 2170, 294, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0]])
    }

    build_data_y = tf.constant([1, 1, 0, 0, 1])

    ########################################################################
    # Feature params
    ########################################################################
    # 1 params
    # 1.1 feature embedding
    feature_embedding_dim = 16
    attention_embedding_size = 16
    dense_units = 16
    cross_op_layer_vec_emb_size = 8

    # 1.2 other params
    # sparse
    pr_vs = 1037
    ci_vs = 7116
    # varlen
    co_vs = 87
    ac_vs = 73994
    di_vs = 17127
    # 1.3
    st_vs = 4973

    # 2 Features
    # 2.1 User features
    # 2.1.1 SparseFeat
    user_sparse_feature_columns = [
        SparseFeat("u_pr", vocabulary_size=pr_vs, embedding_dim=feature_embedding_dim),
        SparseFeat("u_ci", vocabulary_size=ci_vs, embedding_dim=feature_embedding_dim),
    ]

    # 2.1.2 VarLenSparseFeat
    user_varlen_feature_columns = [
        VarLenSparseFeat(
            SparseFeat('u_co', vocabulary_size=co_vs, embedding_dim=feature_embedding_dim),
            maxlen=3,
            combiner='mask_to_zero',
        ),
        VarLenSparseFeat(
            SparseFeat('u_ac', vocabulary_size=ac_vs, embedding_dim=feature_embedding_dim),
            maxlen=25,
            combiner='mask_to_zero',
        ),
        VarLenSparseFeat(
            SparseFeat('u_di', vocabulary_size=di_vs, embedding_dim=feature_embedding_dim),
            maxlen=5,
            combiner='mask_to_zero',
        ),
    ]

    # 2.1.3 tags in user features
    user_varlen_tag_columns = [
        VarLenSparseFeat(
            SparseFeat('u_st', vocabulary_size=st_vs, embedding_dim=feature_embedding_dim,
                       embedding_name="small_type"),
            maxlen=30,
            combiner='mask_to_zero',
        ),
    ]

    # 2.2 item feature - tag whose score to be predicted
    item_varlen_tag_column = [
        VarLenSparseFeat(
            SparseFeat('i_st', vocabulary_size=st_vs, embedding_dim=feature_embedding_dim,
                       embedding_name="small_type"),
            maxlen=15,
            combiner='mask_to_zero'),
    ]
