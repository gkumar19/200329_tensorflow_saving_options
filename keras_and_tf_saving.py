# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:05:34 2020

@author: KGU2BAN
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

#%%create a simple model
input_x = Input(shape=(1,), name='input_layer0')
x = Dense(1,activation = 'linear', name='dense_layer0')(input_x)
model = Model(inputs=input_x, outputs=x, name='1layer_model')
model.summary()
model.compile(Adam(0.1), 'mse', metrics=['mae'])
train_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
train_y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
model.fit(train_x, train_y, epochs=10000)
print(model.predict([20, 99]))

#%%usecase1: use keras .h5 file to save model for later usage in coding with saved weights
tf.keras.models.save_model(model, 'model_keras.h5')
model2 = tf.keras.models.load_model('model_keras.h5')

model2.summary()

#%%usecase2: use tf file to save model for usage in serving
tf.saved_model.save(model, 'model_tf0')
model2_tf = tf.saved_model.load('model_tf0')

print(list(model2_tf.signatures.keys()))
infer_model = model2_tf.signatures['serving_default']
infer_model.structured_input_signature
infer_model.structured_outputs
predict = infer_model(input_layer0 = tf.constant(np.array([1, 2])[:,None], dtype='float32'))['dense_layer0'].numpy()
