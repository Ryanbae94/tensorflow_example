import tensorflow as tf
import numpy as np
from tensorflow import keras


def house_model(y_new):
    xs = np.array([1, 2, 3, 4, 5, 8, 12], dtype=float)
    ys = np.array([100, 150, 200, 250, 300, 450, 650], dtype=float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer = 'sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500)
    return model.predict(y_new)[0]


prediction = house_model([7.0])
print(prediction)
