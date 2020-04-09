import tensorflow as tf
import numpy as np
from crf import CRF  
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    gpu_num = len(gpus)
    if gpu_num > 1:
        strategy = tf.distribute.MirroredStrategy(
            devices=['/gpu:{}'.format(i) for i in range(gpu_num)]
        )
    else:
        strategy = None
    
    #with strategy.scope():
    x = tf.keras.layers.Input(shape=(10,), dtype=tf.int32)
    e = tf.keras.layers.Embedding(1000, 128, mask_zero=True)(x)
    h = tf.keras.layers.GRU(128, return_sequences=True)(e)
    crf = CRF(10, sparse_target=True, lr_multiplier=10, name='crf_layer')
    y = crf(h)
    
    model = tf.keras.Model(x, y)
    model.compile(optimizer='adam', loss=crf.loss, metrics=[crf.accuracy])

    import numpy as np
    xx = np.random.randint(1, 999, (1000, 10))
    xx[:, 6:] = 0
    yy = np.random.randint(0, 9, (1000, 10, 1))
    
    model.fit(xx, yy, batch_size=64, epochs=10)
    
    print(model.predict(xx)[0].argmax(-1))