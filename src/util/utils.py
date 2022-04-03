import sys
from collections import OrderedDict
import tensorflow as tf
from keras.layers import Input, Embedding
from typing import List, Dict

from tensorflow.python.keras.layers import Dense

from util.feature_columns import SparseFeat, VarLenSparseFeat
from model.layer import SequencePoolingLayerK, WeightComputeLayer
from model.layer import AttentionUTPM


def create_attention_first_layer_head_dict(head_rank: int,
                                           var_feat_cols: List[SparseFeat or VarLenSparseFeat],
                                           att_dense_units: int = 8):
    attenion_first_head_dict = OrderedDict()

    for feat in var_feat_cols:
        feat_name = feat.name
        attenion_first_head_dict[feat_name] = Dense(att_dense_units,
                                                    activation="relu",
                                                    name=feat_name + "_first_layer_dense_head_{}".format(head_rank))

    return attenion_first_head_dict


def create_inputs_dict(feat_cols: List[SparseFeat or VarLenSparseFeat]):
    feat_input_dict = OrderedDict()

    for feat in feat_cols:
        if isinstance(feat, SparseFeat):
            feat_input_dict[feat.name] = Input(shape=[1], name=feat.name, dtype=tf.int32)
        elif isinstance(feat, VarLenSparseFeat):
            feat_input_dict[feat.name] = Input(shape=[feat.maxlen], name=feat.name, dtype=tf.int32)
            weight_name = feat.weight_name
            if weight_name:
                feat_input_dict[weight_name] = Input(shape=[feat.maxlen], name=weight_name, dtype=tf.float32)

    return feat_input_dict


def create_embedding_dict(feat_cols: List[SparseFeat or VarLenSparseFeat]):
    embedding_dict = OrderedDict()

    for feat in feat_cols:
        if feat.embedding_name not in embedding_dict:
            if isinstance(feat, SparseFeat):
                embedding_dict[feat.embedding_name] = Embedding(input_dim=feat.vocabulary_size,
                                                                output_dim=feat.embedding_dim,
                                                                name="emb_" + feat.embedding_name)
            elif isinstance(feat, VarLenSparseFeat):
                embedding_dict[feat.embedding_name] = Embedding(input_dim=feat.vocabulary_size,
                                                                output_dim=feat.embedding_dim,
                                                                name="emb_" + feat.embedding_name,
                                                                mask_zero=True)
    return embedding_dict


def sparse_embedding_lookup(embedding_dict: Dict[str, VarLenSparseFeat or SparseFeat],
                            inputs_dict: Dict[str, Input],
                            sparse_feature_columns: List[VarLenSparseFeat or SparseFeat]):
    return_embedding_list = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        lookup_idx = inputs_dict[feature_name]
        return_embedding_list.append(embedding_dict[embedding_name](lookup_idx))

    return return_embedding_list


def varlen_embedding_lookup_dict(embedding_dict: Dict[str, Embedding],
                                 inputs_dict: Dict[str, Input],
                                 varlen_feature_columns: List[VarLenSparseFeat or SparseFeat]):
    embedding_emb_data_dict = OrderedDict()
    for fc in varlen_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        lookup_idx = inputs_dict[feature_name]
        embedding_emb_data_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)

    return embedding_emb_data_dict


def varlen_embedding_lookup_dict_v2(embedding_dict: Dict[str, Embedding],
                                    inputs_dict: Dict[str, Input],
                                    varlen_feature_columns: List[VarLenSparseFeat or SparseFeat],
                                    weight_compute_layer: WeightComputeLayer):
    embedding_emb_data_dict = OrderedDict()
    for fc in varlen_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        weight_name = fc.weight_name
        lookup_idx = inputs_dict[feature_name]

        emb_data = embedding_dict[embedding_name](lookup_idx)

        if weight_name:
            feat_weights = inputs_dict[weight_name]
            emb_data = weight_compute_layer([emb_data, feat_weights])

        embedding_emb_data_dict[feature_name] = emb_data

    return embedding_emb_data_dict


def get_varlen_ori_pooling_list(varlen_embedding_data_dict: Dict[str, Embedding],
                                varlen_feature_columns: List[VarLenSparseFeat or SparseFeat],
                                combiner_input: str,
                                ori_pooling_layer: SequencePoolingLayerK):
    results_list = []
    for fc in varlen_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        if combiner == combiner_input:
            seq_input = varlen_embedding_data_dict[feature_name]
            vec = ori_pooling_layer(seq_input)
            results_list.append(vec)
    return results_list


# weight_name
def get_varlen_weight_pooling_dict(varlen_embedding_data_dict: Dict[str, Embedding],
                                   varlen_feature_columns: List[VarLenSparseFeat or SparseFeat],
                                   combiner_input: str,
                                   pooling_layer_origin: SequencePoolingLayerK):
    results_dict = OrderedDict()
    for fc in varlen_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner

        if combiner_input == combiner:
            seq_embed_list = varlen_embedding_data_dict[feature_name]
            vec = pooling_layer_origin(seq_embed_list)
            results_dict[feature_name] = vec

    return results_dict


# weight_name
def get_first_layer_att_result_list(user_varlen_feat_pool_dict: dict,
                                    att_first_head_dict: dict,
                                    feat_cols: List[VarLenSparseFeat or SparseFeat],
                                    attention_unit: AttentionUTPM):
    result_list = []
    for feat in feat_cols:
        feat_name = feat.name
        emb_data = user_varlen_feat_pool_dict[feat_name]
        att_outputs = att_first_head_dict[feat_name](emb_data)
        result = attention_unit([emb_data, att_outputs])
        result_list.append(result)

    return result_list
