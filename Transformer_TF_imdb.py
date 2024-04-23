from keras.utils import pad_sequences
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
import Transformer_TF

# IMDb 데이터셋 로드
max_features = 20000
maxlen = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# 패딩 추가
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# 데이터셋 분할
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# 하이퍼파라미터 설정
vocab_size = max_features
num_layers = 6
dff = 512
d_model = 256
num_heads = 8
dropout = 0.1

# 모델 생성
transformer = Transformer_TF.transformer(
    vocab_size=vocab_size,
    num_layers=num_layers,
    dff=dff,
    d_model=d_model,
    num_heads=num_heads,
    dropout=dropout
)

# 모델 컴파일
transformer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
history = transformer.fit(
    [X_train, X_train], y_train,
    epochs=10,
    batch_size=64,
    validation_data=([X_val, X_val], y_val)
)

# 모델 평가
test_loss, test_acc = transformer.evaluate([X_test, X_test], y_test)
print("Test Accuracy:", test_acc)