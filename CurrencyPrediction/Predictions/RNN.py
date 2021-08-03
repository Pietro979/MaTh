import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None,1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10)
])