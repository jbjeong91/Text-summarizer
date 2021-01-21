# -*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_string('cmd', 'test', 'cmd')  # train or test
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')  # 배치 크기
tf.app.flags.DEFINE_integer('train_steps', 10000, 'train steps')  # 학습 에포크
tf.app.flags.DEFINE_float('dropout_width', 0.9, 'dropout width')  # 드롭아웃 크기
tf.app.flags.DEFINE_integer('embedding_size', 128, 'embedding size')  # 가중치 크기 # 논문 512 사용
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')  # 학습률
tf.app.flags.DEFINE_integer('shuffle_seek', 1000, 'shuffle random seek')  # 셔플 시드값
tf.app.flags.DEFINE_integer('max_sequence_length', 60, 'max sequence length')  # 시퀀스 길이
tf.app.flags.DEFINE_integer('model_hidden_size', 128, 'model weights size')  # 모델 가중치 크기
tf.app.flags.DEFINE_integer('ffn_hidden_size', 512, 'ffn weights size')  # ffn 가중치 크기
tf.app.flags.DEFINE_integer('attention_head_size', 8, 'attn head size')  # 멀티 헤드 크기
tf.app.flags.DEFINE_integer('layer_size', 4, 'layer size')
tf.app.flags.DEFINE_string('data_path', './data_in/Reviews.csv', 'data path')  # 아마존 리뷰 데이터
tf.app.flags.DEFINE_string('vocabulary_path', './data_out/vocabularyData.voc', 'vocabulary path')  # 사전 위치
tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point', 'check point path')  # 체크 포인트 위치
#tf.app.flags.DEFINE_string('check_point_path', './data_out/check_point2', 'check point path')  # 체크 포인트 위치
tf.app.flags.DEFINE_boolean('tokenize_as_WordPunctTokenizer', True, 'set WordPunctTokenizer')
tf.app.flags.DEFINE_boolean('xavier_initializer', True, 'set xavier initializer')  # xavier initializer를 사용할 것인지에 대한 

# Define FLAGS
DEFINES = tf.app.flags.FLAGS
