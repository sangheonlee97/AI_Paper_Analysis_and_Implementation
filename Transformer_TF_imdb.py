import numpy as np
from keras.utils import pad_sequences
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
import Transformer_TF_m
import tensorflow as tf
import matplotlib.pyplot as plt

# IMDb 데이터셋 로드
max_features = 10000
maxlen = 100
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# 패딩 추가
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# 데이터셋 분할
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
# print(X_train.shape)
# print(np.unique(y_train, return_counts=True))
# 하이퍼파라미터 설정
vocab_size = max_features
num_layers = 4
dff = 512
d_model = 128
num_heads = 4
dropout = 0.1

# 모델 생성
model = Transformer_TF_m.transformer(
    vocab_size=vocab_size,
    num_layers=num_layers,
    dff=dff,
    d_model=d_model,
    num_heads=num_heads,
    dropout=dropout,
    name='tftf'
)


def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, maxlen - 1))

  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)
     

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
     

sample_learning_rate = CustomSchedule(d_model=d_model)

plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")

learning_rate = CustomSchedule(d_model=d_model)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
  # ensure labels have shape (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, maxlen))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 모델 컴파일
model.summary()

# 모델 학습
history = model.fit(
    [X_train, X_train], y_train,
    epochs=10,
    batch_size=64,
    validation_data=([X_val, X_val], y_val)
)

# 모델 평가
test_loss, test_acc = model.evaluate([X_test, X_test], y_test)
print("Test Accuracy:", test_acc)