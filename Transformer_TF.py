import tensorflow as tf
from keras import layers
import numpy as np

def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    d_k = tf.cast(tf.shape(key)[-1], tf.floate32)   # key의 마지막 차원의 크기를 float32로 변환
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)
    output = tf.matmul(attention_weights, value)
    return output, attention_weights
