# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
from configs import DEFINES
import numpy as np
tf.enable_eager_execution

def layer_norm(inputs, eps=1e-6):
    # LayerNorm(x + Sublayer(x))
    feature_shape = inputs.get_shape()[-1:]
    #  평균과 표준편차을 넘겨 준다.
    mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)
    std = tf.keras.backend.std(inputs, [-1], keepdims=True)
    beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
    gamma = tf.Variable(tf.ones(feature_shape), trainable=False)

    return gamma * (inputs - mean) / (std + eps) + beta

def sublayer_connection(inputs, sublayer, dropout=0.2):
    # LayerNorm(x + Sublayer(x))
    outputs = layer_norm(inputs + tf.keras.layers.Dropout(dropout)(sublayer))
    return outputs

def feed_forward(inputs, num_units):
    # FFN(x) = max(0, xW1 + b1)W2 + b2
    feature_shape = inputs.get_shape()[-1]
    inner_layer = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(feature_shape)(inner_layer)

    return outputs

def positional_encoding(dim, sentence_length):
    # Positional Encoding
    # paper: https://arxiv.org/abs/1706.03762
    # P E(pos,2i) = sin(pos/100002i/dmodel)
    # P E(pos,2i+1) = cos(pos/100002i/dmodel)
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim)
                            for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=tf.float32)

#수정됨
def scaled_dot_product_attention(query, key, value, masked=False):
    # Attention(Q, K, V ) = softmax(QKt / root dk)V
    key_dim_size = float(key.get_shape().as_list()[-1])
    key = tf.transpose(key, perm=[0, 2, 1])
    outputs = tf.matmul(query, key) / tf.sqrt(key_dim_size)
    #print('여기체크 output1',outputs)

    if masked:
        diag_vals = tf.ones_like(outputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

        paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    #print('여기체크 output2', outputs)
    attention_map = tf.nn.softmax(outputs)

    return tf.matmul(attention_map, value), attention_map

#수정됨
def multi_head_attention(query, key, value, num_units, heads, masked=False):
    query = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(query)
    key = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(key)
    value = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(value)

    query = tf.concat(tf.split(query, heads, axis=-1), axis=0)
    key = tf.concat(tf.split(key, heads, axis=-1), axis=0)
    value = tf.concat(tf.split(value, heads, axis=-1), axis=0)
    attention_map, attn_weight = scaled_dot_product_attention(query, key, value, masked)
    attn_outputs = tf.concat(tf.split(attention_map, heads, axis=0), axis=-1)

    attn_outputs = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(attn_outputs)
    attn_weight = tf.expand_dims(attn_weight, axis=1)
    attn_weight = tf.concat(tf.split(attn_weight, heads, axis=0), axis=1)
    print('정범 체크 attn_weight', attn_weight) # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    return attn_outputs, attn_weight

#수정됨
def encoder_module(inputs, model_dim, ffn_dim, heads):
    attn_output, _ = multi_head_attention(inputs, inputs, inputs, model_dim, heads)
    self_attn = sublayer_connection(inputs, attn_output)
    outputs = sublayer_connection(self_attn, feed_forward(self_attn, ffn_dim))
    return outputs

def decoder_module(inputs, encoder_outputs, model_dim, ffn_dim, heads):
    attn_output, attn_weight_block1 = multi_head_attention(inputs, inputs, inputs, model_dim, heads, masked=True)
    masked_self_attn = sublayer_connection(inputs, attn_output)
    ''' #원본
    self_attn = sublayer_connection(masked_self_attn, multi_head_attention(masked_self_attn, encoder_outputs,
                                                                           encoder_outputs, model_dim, heads))
    outputs = sublayer_connection(self_attn, feed_forward(self_attn, ffn_dim))
    return outputs
    '''
    attn_output, attn_weight_block2 = multi_head_attention(masked_self_attn, encoder_outputs, encoder_outputs, model_dim, heads)
    self_attn = sublayer_connection(masked_self_attn, attn_output)
    outputs = sublayer_connection(self_attn, feed_forward(self_attn, ffn_dim))
    return outputs, attn_weight_block1, attn_weight_block2

def encoder(inputs, model_dim, ffn_dim, heads, num_layers):
    outputs = inputs
    for i in range(num_layers):
        outputs = encoder_module(outputs, model_dim, ffn_dim, heads)

    return outputs

def decoder(inputs, encoder_outputs, model_dim, ffn_dim, heads, num_layers):
    outputs = inputs

    attention_weights = {}
    for i in range(num_layers):
        outputs, block1, block2 = decoder_module(outputs, encoder_outputs, model_dim, ffn_dim, heads)
        attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

    print('#정범 체크attn_weights', attention_weights)
    print('정범 체크block2',block2)
    # context vectors
    enc_out_shape = tf.shape(encoder_outputs)
    context = tf.reshape(encoder_outputs, (enc_out_shape[0], enc_out_shape[1], heads, int(model_dim/heads)))  # shape : (batch_size, input_seq_len, num_heads, depth)
    context = tf.transpose(context, [0, 2, 1, 3])  # (batch_size, num_heads, input_seq_len, depth)
    context = tf.expand_dims(context, axis=2)  # (batch_size, num_heads, 1, input_seq_len, depth)

    attn = tf.expand_dims(block2, axis=-1)  # (batch_size, num_heads, target_seq_len, input_seq_len, 1)

    context = context * attn  # (batch_size, num_heads, target_seq_len, input_seq_len, depth) #정범 체크context Tensor("mul_49:0", shape=(32, 8, 130, 130, 64), dtype=float32)
    print('#정범 체크context', context)
    context = tf.reduce_sum(context, axis=3)  # (batch_size, num_heads, target_seq_len, depth)
    context = tf.transpose(context, [0, 2, 1, 3])  # (batch_size, target_seq_len, num_heads, depth)
    context = tf.reshape(context, (tf.shape(context)[0], tf.shape(context)[1], model_dim))  # (batch_size, target_seq_len, d_model)

    Wx = tf.keras.layers.Dense(1)
    Ws = tf.keras.layers.Dense(1)
    Wh = tf.keras.layers.Dense(1)

    V = tf.keras.layers.Dense(1)
    p_gen = tf.sigmoid(Wx(inputs) + Ws(outputs) + Wh(context))

    return outputs, attention_weights, p_gen


def _PGN(x, gens, vocab_dists, attn_dists, vocab_size):

    with tf.variable_scope('final_distribution', reuse=tf.AUTO_REUSE):
        # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
        vocab_dists = [p_gen * dist for (p_gen, dist) in zip(gens, vocab_dists)]
        attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(gens, attn_dists)]
        batch_size = tf.shape(x)[0]
        dec_t = tf.shape(attn_dists)[1]
        attn_len = tf.shape(attn_dists)[2]

        batch_num = tf.range(0, limit=batch_size)
        batch_num = tf.expand_dims(batch_num,1)
        batch_num = tf.tile(batch_num, [1, attn_len])
        batch_num = tf.cast(batch_num, tf.int32)
        indices = tf.stack((batch_num, x), axis=2)
        shape = [batch_size, vocab_size]
        attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]  # list length max_dec_steps (batch_size, extended_vsize)

        final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_dists, attn_dists_projected)]

    return final_dists


def _PGN_loss(final_dists, targets):

    with tf.name_scope('loss'):
        dec = tf.shape(targets)[1] # seq 길이
        batch_nums = tf.shape(targets)[0]
        dec = tf.range(0, limit=dec)
        dec = tf.expand_dims(dec, axis=0)
        dec = tf.tile(dec, [batch_nums, 1])
        dec = tf.cast(dec, tf.int32)
        indices = tf.stack([dec, targets], axis=2)  # [batch_size, dec, 2]

        loss = tf.map_fn(fn=lambda x: tf.gather_nd(x[1], x[0]), elems=(indices, final_dists), dtype=tf.float32)
        loss = tf.log(0.9) - tf.log(loss + (1e-10))

        nonpadding = tf.to_float(tf.not_equal(targets, 0))  # 0: <pad>
        loss = tf.reduce_sum(loss * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        return loss


def Model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    position_encode = positional_encoding(params['embedding_size'], params['max_sequence_length'])

    if params['xavier_initializer']:
        embedding_initializer = 'glorot_normal'
    else:
        embedding_initializer = 'uniform'

    embedding = tf.keras.layers.Embedding(params['vocabulary_length'],
                                          params['embedding_size'],
                                          embeddings_initializer=embedding_initializer)

    x_embedded_matrix = embedding(features['input']) + position_encode
    y_embedded_matrix = embedding(features['output']) + position_encode

    encoder_outputs = encoder(x_embedded_matrix, params['model_hidden_size'], params['ffn_hidden_size'],
                              params['attention_head_size'], params['layer_size'])
    # 수정된 부분----------------
    decoder_outputs, attn_weight, p_gen = decoder(y_embedded_matrix, encoder_outputs, params['model_hidden_size'], params['ffn_hidden_size'], params['attention_head_size'], params['layer_size'])
    #-------------------------

    # 추가된 부분----------------
    logits_temp = tf.keras.layers.Dense(params['vocabulary_length'])(decoder_outputs)
    output = tf.nn.softmax(logits_temp)

    attn_dists = attn_weight['decoder_layer{}_block2'.format(params['layer_size'])]  # (batch_size,num_heads, targ_seq_len, inp_seq_len)
    attn_dists = tf.reduce_sum(attn_dists, axis=1) / params['attention_head_size']  # (batch_size, targ_seq_len, inp_seq_len)
    final_dists = _PGN(features['input'], tf.unstack(p_gen, axis=1), tf.unstack(output, axis=1), tf.unstack(attn_dists, axis=1), params['vocabulary_length'])
    logits = tf.stack(final_dists, axis=1)
    #------------------------

    #logits = tf.keras.layers.Dense(params['vocabulary_length'])(decoder_outputs)

    predict = tf.argmax(logits, 2)

    if PREDICT:
        predictions = {
            'indexs': predict,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # 정답 차원 변경을 한다. [배치 * max_sequence_length * vocabulary_length]  
    # logits과 같은 차원을 만들기 위함이다.

    #수정된 loss---------
    labels_ = tf.one_hot(labels, params['vocabulary_length'])
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_))
    loss2 = _PGN_loss(final_dists=final_dists, targets=labels)
    loss = loss1 + loss2
    #-------------------
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict)

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert TRAIN

    # lrate = d−0.5 *  model · min(step_num−0.5, step_num · warmup_steps−1.5)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
