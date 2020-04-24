# CRF layer for tf2.x

## Requirements

tensorflow/tensorflow_gpu >= 2.0.0



## Example

```python
from crf import CRF
import tensorflow as tf

# build model
x = tf.keras.layers.Input(shape=(10,))
e = tf.keras.layers.Embedding(1000, 128, mask_zero=True)(x)
h = tf.keras.layers.LSTM(128, return_sequences=True)(e)
crf = CRF(15, sparse_target=True)
y = crf(h)
model = tf.keras.Model(x, y)
model.summary()

# train model
model.compile(
	optimizer='adam',
	loss=crf.loss,
	metrics=[crf.accuracy],
)
model.fit(x_train, y_train)

# infer
logits = model(x_test)
y_pred = crf.decode(logits)
```

