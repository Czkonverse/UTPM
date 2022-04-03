from collections import OrderedDict

from keras.layers import Concatenate, Flatten, Dense
from keras import Model

from model.layer import SequencePoolingLayerK, WeightComputeLayer
from util.utils import create_inputs_dict, create_embedding_dict, sparse_embedding_lookup, \
    varlen_embedding_lookup_dict, create_attention_first_layer_head_dict, varlen_embedding_lookup_dict_v2, \
    get_varlen_weight_pooling_dict, get_varlen_ori_pooling_list, get_first_layer_att_result_list
from model.layer import LinerOpLayerUTPM, CrossOpLayerUTPM, JointLossUTPM, AttentionUTPM


def UTPMFunc(
        user_sparse_feature_columns,
        user_varlen_feature_columns,
        user_varlen_tag_columns,
        item_varlen_tag_column,
        cross_op_layer_vec_emb_size,
        feature_embedding_dim=16,
        attention_embedding_size=16,
        dense_units=64
):
    # 1 Input layer
    all_feat_cols = user_sparse_feature_columns + user_varlen_feature_columns + user_varlen_tag_columns + item_varlen_tag_column
    norm_user_feat_cols = user_sparse_feature_columns + user_varlen_feature_columns

    inputs_dict = create_inputs_dict(all_feat_cols)

    # 2 Model Structure
    # 2.1 Embedding structure
    # 2.1.1 norm_user_feat_cols embedding
    norm_user_embedding_dict = create_embedding_dict(norm_user_feat_cols)

    # 2.1.2 tag embedding - user and item feat share this embedding
    tag_embedding_dict = create_embedding_dict(item_varlen_tag_column)

    # 2.2 attention first layer
    # 2.2.1 head 1
    att_first_layer_head_1_dict = create_attention_first_layer_head_dict(
        head_rank=1,
        var_feat_cols=user_varlen_feature_columns + user_varlen_tag_columns,
        att_dense_units=attention_embedding_size)
    att_second_dense_head_1 = Dense(attention_embedding_size, activation="relu", name="att_second_dense_head_1")

    # 2.2.2 head 2
    att_first_layer_head_2_dict = create_attention_first_layer_head_dict(
        head_rank=2,
        var_feat_cols=user_varlen_feature_columns + user_varlen_tag_columns,
        att_dense_units=attention_embedding_size)
    att_second_dense_head_2 = Dense(attention_embedding_size, activation="relu", name="att_second_dense_head_2")

    # 2.3 Varlen poolinge layer
    pooling_layer_original = SequencePoolingLayerK(name="pooling_layer_original", mode="mask_to_zero")
    weight_compute_layer = WeightComputeLayer(name="weight_compute_layer")

    # 2.4 Common function layer
    concat_layer_axis_1 = Concatenate(axis=1, name="concat_layer_axis_1")
    flatten_layer = Flatten(name="flatten_layer_1")
    dense_layer_linear_op = Dense(dense_units, activation='relu', name="dense_layer_linear_op")
    dense_layer_cross_op = Dense(dense_units, activation='relu', name="dense_layer_cross_op")
    dense_layer_concat_two_op = Dense(feature_embedding_dim, activation='relu', name="dense_layer_concat_two_op")

    # 2.5 UTPM special layers
    att_unit_layer_head_1 = AttentionUTPM(att_layer_num="first", att_dense_unit=attention_embedding_size,
                                          name="att_head_1_att_unit")
    att_unit_layer_head_2 = AttentionUTPM(att_layer_num="second", att_dense_unit=attention_embedding_size,
                                          name="att_head_2_att_unit")

    linear_op_layer = LinerOpLayerUTPM(name="linear_op_layer")

    cross_op_layer = CrossOpLayerUTPM(feature_embedding_dim=feature_embedding_dim,
                                      cross_vec_emb_size=cross_op_layer_vec_emb_size,
                                      name="cross_op_layer")

    # 2.6 most special layer - "if training: "
    joint_loss_layer = JointLossUTPM(name="joint_loss_layer")  # check the output dimension

    # 3 Computation
    # 3.1 Prepare embedding data
    # 3.1.1 user sparse feature
    user_sparse_feat_emb_list = sparse_embedding_lookup(norm_user_embedding_dict,
                                                        inputs_dict,
                                                        user_sparse_feature_columns)
    # 3.1.2 user varlen feat - except tags
    user_varlen_feat_emb_data_dict = varlen_embedding_lookup_dict_v2(norm_user_embedding_dict,
                                                                     inputs_dict,
                                                                     user_varlen_feature_columns,
                                                                     weight_compute_layer)
    user_varlen_not_tag_feat_pool_dict = get_varlen_weight_pooling_dict(user_varlen_feat_emb_data_dict,
                                                                        user_varlen_feature_columns,
                                                                        "mask_to_zero",
                                                                        pooling_layer_original)
    # 3.1.3 user varlen feat - tag
    user_varlen_tag_feat_emb_data_dict = varlen_embedding_lookup_dict_v2(tag_embedding_dict,
                                                                         inputs_dict,
                                                                         user_varlen_tag_columns,
                                                                         weight_compute_layer)
    user_varlen_tag_feat_pool_dict = get_varlen_weight_pooling_dict(user_varlen_tag_feat_emb_data_dict,
                                                                    user_varlen_tag_columns,
                                                                    "mask_to_zero",
                                                                    pooling_layer_original)

    # 3.1.4 item - tag - special, the mask data have been set to zero, but the output dimension was not changed.
    item_varlen_tag_emb_data_dict = varlen_embedding_lookup_dict(tag_embedding_dict,
                                                                 inputs_dict,
                                                                 item_varlen_tag_column)
    item_tag_varlen_feat_emb_data_list = get_varlen_ori_pooling_list(item_varlen_tag_emb_data_dict,
                                                                     item_varlen_tag_column,
                                                                     "mask_to_zero",
                                                                     pooling_layer_original)
    item_tag_emb_data = item_tag_varlen_feat_emb_data_list[0]

    # 3.2 Specific head computation
    # 3.2.1 concat varlen feat pooling data dict into one dict
    user_varlen_feat_pool_dict = OrderedDict()
    user_varlen_feat_pool_dict.update(user_varlen_not_tag_feat_pool_dict)
    user_varlen_feat_pool_dict.update(user_varlen_tag_feat_pool_dict)

    # 3.2.3 attention - first layer
    # 3.2.3.1 head 1
    first_layer_output_head_1_list = get_first_layer_att_result_list(user_varlen_feat_pool_dict,
                                                                     att_first_layer_head_1_dict,
                                                                     user_varlen_feature_columns + user_varlen_tag_columns,
                                                                     att_unit_layer_head_1)
    # 3.2.3.2 head 2
    first_layer_output_head_2_list = get_first_layer_att_result_list(user_varlen_feat_pool_dict,
                                                                     att_first_layer_head_2_dict,
                                                                     user_varlen_feature_columns + user_varlen_tag_columns,
                                                                     att_unit_layer_head_2)

    # 3.2.4 attention - second layer
    # 3.2.4.1 head_1
    first_layer_output_head_1 = concat_layer_axis_1(user_sparse_feat_emb_list + first_layer_output_head_1_list)
    first_layer_output_att_score_head_1 = att_second_dense_head_1(first_layer_output_head_1)
    scend_layer_output_head_1 = att_unit_layer_head_1([first_layer_output_head_1, first_layer_output_att_score_head_1])
    flat_output_head_1 = flatten_layer(scend_layer_output_head_1)

    # 3.2.4.2 head_2
    first_layer_output_head_2 = concat_layer_axis_1(user_sparse_feat_emb_list + first_layer_output_head_2_list)
    first_layer_output_att_score_head_2 = att_second_dense_head_2(first_layer_output_head_2)
    scend_layer_output_head_2 = att_unit_layer_head_2([first_layer_output_head_2, first_layer_output_att_score_head_2])
    flat_output_head_2 = flatten_layer(scend_layer_output_head_2)

    # 3.3 Cross feature layer
    concat_data = concat_layer_axis_1([flat_output_head_1, flat_output_head_2])
    # 3.3.1 linear op
    linear_op_result = linear_op_layer(concat_data)
    # 3.3.2 cross op
    cross_op_result = cross_op_layer(concat_data)

    # 3.4 Fully connect layer
    fc_linear_result = dense_layer_linear_op(linear_op_result)
    fc_cross_result = dense_layer_cross_op(cross_op_result)
    concat_two_fc = concat_layer_axis_1([fc_linear_result, fc_cross_result])

    # 3.5 user embeddings
    user_embeds = dense_layer_concat_two_op(concat_two_fc)

    # 4 special layers
    final_output = joint_loss_layer([user_embeds, item_tag_emb_data])

    # 5 output
    model = Model(inputs=inputs_dict, outputs=final_output)

    model.__setattr__("inputs_dict", inputs_dict)
    model.__setattr__("user_embeds", user_embeds)

    return model


class UTPMFuncUtil:
    @classmethod
    def get_model(cls,
                  user_sparse_feature_columns,
                  user_varlen_feature_columns,
                  user_varlen_tag_columns,
                  item_varlen_tag_column,
                  cross_op_layer_vec_emb_size,
                  feature_embedding_dim=16,
                  attention_embedding_size=16,
                  dense_units=64
                  ):
        return UTPMFunc(user_sparse_feature_columns,
                              user_varlen_feature_columns,
                              user_varlen_tag_columns,
                              item_varlen_tag_column,
                              cross_op_layer_vec_emb_size,
                              feature_embedding_dim=feature_embedding_dim,
                              attention_embedding_size=attention_embedding_size,
                              dense_units=dense_units
                              )

    @classmethod
    def get_customer_objects(cls):
        custom_objects = {
            "SequencePoolingLayerK": SequencePoolingLayerK,
            "WeightComputeLayer": WeightComputeLayer,
            "AttentionUTPM": AttentionUTPM,
            "LinerOpLayerUTPM": LinerOpLayerUTPM,
            "CrossOpLayerUTPM": CrossOpLayerUTPM,
            "JointLossUTPM": JointLossUTPM,
        }
        return custom_objects
