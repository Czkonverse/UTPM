import tensorflow as tf

from model.UTPMFunc import UTPMFunc
from keras.losses import binary_crossentropy
from keras.metrics import AUC, BinaryAccuracy, Precision, Recall
from util.config import Config

if __name__ == '__main__':
    # 0 Config
    config = Config()

    # 1 Model
    model = UTPMFunc(
        config.user_sparse_feature_columns,
        config.user_varlen_feature_columns,
        config.user_varlen_tag_columns,
        config.item_varlen_tag_column,
        config.cross_op_layer_vec_emb_size,
        feature_embedding_dim=config.feature_embedding_dim,
        attention_embedding_size=config.attention_embedding_size,
        dense_units=config.dense_units,
    )

    # 2 Train
    # 2.1 - Compile
    model.compile(
        loss=binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=[AUC(name='auc'), BinaryAccuracy('binary_acc'), Precision(name='precision'), Recall(name='recall'), ]
    )

    # 2.2 fit
    model.fit(
        config.build_data,
        config.build_data_y,
        epochs=1,
        verbose=1,
        validation_data=[config.build_data, config.build_data_y]
    )

    # 2.3 model summary
    model.summary()

    # 3 Export model
    user_embed_model = tf.keras.Model(inputs=model.inputs_dict, outputs=model.user_embeds)
    result = user_embed_model.predict(config.build_data)
    print(result)
