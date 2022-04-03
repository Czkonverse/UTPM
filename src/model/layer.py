import tensorflow as tf
from keras.layers import Layer
from tensorflow.python.keras.layers import Dense


class SequencePoolingLayerK(Layer):
    def __init__(self, mode='sum', supports_masking=True, **kwargs):

        if mode not in ['sum', 'mean', 'max', 'mask_to_zero']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.supports_masking = supports_masking
        self.eps = tf.constant(1e-8, tf.float32)
        super(SequencePoolingLayerK, self).__init__(**kwargs)

    def build(self, input_shape):
        if not self.supports_masking:
            raise ValueError("the varlen embedding should self.supports_masking=True.")
        super(SequencePoolingLayerK, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):

        if mask is None:
            raise ValueError("When supports_masking=True,input must support masking")
        uiseq_embed_list = seq_value_len_list
        mask = tf.cast(mask, tf.float32)  # tf.to_float(mask)
        user_behavior_length = tf.reduce_sum(mask, axis=-1, keepdims=True)
        mask = tf.expand_dims(mask, axis=2)

        embedding_size = uiseq_embed_list.shape[-1]

        mask = tf.tile(mask, [1, 1, embedding_size])

        if self.mode == "mask_to_zero":
            hist = uiseq_embed_list * mask
            return hist

        if self.mode == "max":
            hist = uiseq_embed_list - (1 - mask) * 1e9
            return tf.reduce_max(hist, 1, keepdims=True)

        hist = tf.reduce_sum(uiseq_embed_list * mask, 1, keepdims=False)

        if self.mode == "mean":
            hist = tf.divide(hist, tf.cast(user_behavior_length, tf.float32) + self.eps)

        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.mode == "mask_to_zero":
            return (None, input_shape[1], input_shape[2])
        else:
            return (None, 1, input_shape[-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {
            'mode': self.mode,
            'supports_masking': self.supports_masking
        }
        base_config = super(SequencePoolingLayerK, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightComputeLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightComputeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(WeightComputeLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        seq_embed_list = inputs[0]
        feat_weights = inputs[1]

        feat_weights = tf.expand_dims(feat_weights, axis=2)
        outputs = seq_embed_list * feat_weights

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], input_shape[0][2])

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self):
        return super(WeightComputeLayer, self).get_config()


class AttentionUTPM(Layer):

    def __init__(self, att_layer_num, att_dense_unit: int = 8, **kwargs):
        self.att_layer_num = att_layer_num
        self.att_dense_unit = att_dense_unit
        super(AttentionUTPM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.q_vec = self.add_weight(name="attention_weights_" + self.att_layer_num,
                                     shape=[1, self.att_dense_unit],
                                     dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal(seed=1024))

        super(AttentionUTPM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        emb_data = inputs[0]  # None, feature nums, embedding_size
        att_inputs = inputs[1]  # None, feature nums, embedding_size

        #  q * att_inputs
        att_score = self.q_vec * att_inputs
        att_score = tf.reduce_sum(att_score, axis=2)
        att_score = tf.nn.softmax(att_score, axis=1)
        att_score = tf.expand_dims(att_score, axis=2)
        #
        result = emb_data * att_score
        result = tf.reduce_sum(result, axis=1)
        result = tf.expand_dims(result, axis=1)
        return result

    def compute_output_shape(self, input_shape):
        # return input_shape
        return (None, input_shape[1][2])

    def get_config(self):
        config = {
            'att_layer_num': self.att_layer_num,
            'att_dense_unit': self.att_dense_unit,
        }
        base_config = super(AttentionUTPM, self).get_config()
        base_config.update(config)
        return base_config


class PredictTagsLayer(Layer):

    def __init__(self, top_k=20, **kwargs):
        self.top_k = top_k
        super(PredictTagsLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PredictTagsLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        user_embs_raw = inputs[0]
        tags_embedding_table = inputs[1]

        tag_nums = tf.shape(tags_embedding_table)[0]
        user_nums = tf.shape(user_embs_raw)[0]
        feature_embedding_dim = tf.shape(user_embs_raw)[1]

        user_embs = tf.tile(tf.expand_dims(user_embs_raw, axis=1), [1, tag_nums, 1])
        tag_embs = tf.reshape(tf.tile(tags_embedding_table, [user_nums, 1]),
                              [user_nums, tag_nums, feature_embedding_dim])

        score_matrix = tf.reduce_sum(user_embs * tag_embs, axis=-1)

        tag_idx_matrix = tf.argsort(score_matrix, axis=1, direction='DESCENDING')[:, :self.top_k]

        return tag_idx_matrix

    def compute_output_shape(self, input_shape):
        return (None, self.top_k)

    def get_config(self):
        config = {'top_k': self.top_k}
        base_config = super(PredictTagsLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# UTPM - cross operation layer
class CrossOpLayerUTPM(Layer):
    def __init__(self, feature_embedding_dim, cross_vec_emb_size, **kwargs):
        self.input_feat_emb = feature_embedding_dim * 2  # head nums is: 2
        self.cross_vec_emb_size = cross_vec_emb_size
        super(CrossOpLayerUTPM, self).__init__(**kwargs)

    def build(self, input_shape):
        # latent emb matix
        self.latent_emb = self.add_weight(name='latent_emb',
                                          shape=[self.input_feat_emb, self.cross_vec_emb_size],
                                          dtype=tf.float32
                                          )

        # index list of latent embeddings matrix
        index_list_vec = []
        for i in range(self.input_feat_emb - 1):
            for j in range(i + 1, self.input_feat_emb):
                index_list_vec.append(i * self.input_feat_emb + j)

        # index_list_vec = tf.constant(self.index_list)
        self.index_list_vec = tf.expand_dims(index_list_vec, 0)

        super(CrossOpLayerUTPM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]

        inputs_vec = inputs
        inputs_vec = tf.expand_dims(inputs_vec, axis=2)

        # vi * xi
        emb_tmp_1 = tf.multiply(inputs_vec, self.latent_emb)

        emb_tmp_2 = tf.transpose(emb_tmp_1, perm=[0, 2, 1])
        emb_tmp_3 = tf.matmul(emb_tmp_1, emb_tmp_2)
        emb_tmp_4 = tf.reshape(emb_tmp_3, [batch_size, self.input_feat_emb * self.input_feat_emb])

        # final vecs
        self.index_emb = tf.tile(self.index_list_vec, [batch_size, 1])
        concat_vecs = tf.gather(emb_tmp_4, self.index_emb, batch_dims=1)

        return concat_vecs

    def compute_output_shape(self, input_shape):
        output_dim = self.input_feat_emb * (self.input_feat_emb - 1) * 0.5

        return (None, int(output_dim))

    def get_config(self, ):
        config = {
            'input_feat_emb': self.input_feat_emb,
            'cross_vec_emb_size': self.cross_vec_emb_size,
        }
        base_config = super(CrossOpLayerUTPM, self).get_config()
        base_config.update(config)
        return base_config


class LinerOpLayerUTPM(Layer):

    def __init__(self, **kwargs):
        super(LinerOpLayerUTPM, self).__init__(**kwargs)

    def build(self, input_shape):
        vector_size = input_shape[1]

        self.linear_vec = self.add_weight(name='linear_vec',
                                          shape=[1, vector_size],
                                          dtype=tf.float32,
                                          initializer=tf.keras.initializers.TruncatedNormal(seed=1024))

        super(LinerOpLayerUTPM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_value = inputs
        samples_num = tf.shape(inputs)[0]

        linear_matrix = tf.tile(self.linear_vec, [samples_num, 1])

        result = tf.multiply(input_value, linear_matrix)
        return result

    def compute_output_shape(self, input_shape):
        # return input_shape
        return (None, input_shape[1])

    def get_config(self):
        return super(LinerOpLayerUTPM, self).get_config()


class JointLossUTPM(Layer):

    def __init__(self, **kwargs):
        super(JointLossUTPM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_layer_final_output = Dense(1, activation='sigmoid', name="dense_layer_final_output")
        super(JointLossUTPM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        user_embs_raw = inputs[0]

        user_embs = tf.expand_dims(user_embs_raw, axis=1)
        item_embs = inputs[1]

        # tag numbers of each item
        tag_nums = tf.shape(item_embs)[1]

        # user embedding broadcast
        user_embs_tile = tf.tile(user_embs, [1, tag_nums, 1])

        # pair-wise multiply
        result_raw = tf.multiply(user_embs_tile, item_embs)

        result_raw = tf.reduce_sum(tf.reduce_sum(result_raw, axis=2), axis=1, keepdims=True)

        result = self.dense_layer_final_output(result_raw)

        return result

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        return super(JointLossUTPM, self).get_config()


class JointLossUTPMSpecial(Layer):

    def __init__(self, **kwargs):
        super(JointLossUTPMSpecial, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_layer_final_output = Dense(1, activation='sigmoid', name="dense_layer_final_output")
        super(JointLossUTPMSpecial, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        user_embs_raw = inputs[0]

        if training:
            # user_embs
            user_embs = tf.expand_dims(user_embs_raw, axis=1)

            # item_embs
            item_embs_small = inputs[1]
            item_embs_middle = inputs[2]
            item_embs = tf.concat([item_embs_small, item_embs_middle], axis=1)

            # tag numbers of each item
            tag_nums = tf.shape(item_embs)[1]

            # user embedding broadcast
            user_embs_tile = tf.tile(user_embs, [1, tag_nums, 1])

            # pair-wise multiply
            result_raw = tf.multiply(user_embs_tile, item_embs)

            result_raw = tf.reduce_sum(tf.reduce_sum(result_raw, axis=2), axis=1, keepdims=True)

            result = self.dense_layer_final_output(result_raw)
        else:
            result = user_embs_raw

        return result

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        return super(JointLossUTPMSpecial, self).get_config()
