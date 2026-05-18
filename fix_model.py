# fix_model.py mein yeh paste karo
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Purana model load karo
old_model = keras.models.load_model('my_cnn_lstm_model.keras', compile=False)
weights = old_model.get_weights()

# Bilkul same architecture
inputs = keras.Input(shape=(15, 1))
x = keras.layers.Conv1D(32, 3, activation='relu')(inputs)
x = keras.layers.LSTM(32)(x)
x = keras.layers.Dense(16, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

new_model = keras.Model(inputs, outputs)
new_model.set_weights(weights)
new_model.save('my_cnn_lstm_model_v4.h5')
print("Done! v4 saved!")