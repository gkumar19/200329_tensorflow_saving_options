# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 02:16:34 2020

@author: Gaurav
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds

x_train = np.array([1,2,3,4,5,6,7,8, 50, 100])[:,None]
y_train = x_train * 2 - 1

#%%
inputx = Input(shape=(1,) ,name='inputx')
x = Dense(1, name='dense')(inputx)
model = Model(inputx, x)
 
model.compile(tf.keras.optimizers.Adam(0.1), 'mae')
model.fit(x_train, y_train, epochs=100)

model.predict([10])
tf.keras.models.save_model(model, 'keras_saved')

#%%
model_new = tf.saved_model.load('keras_saved')
infer = model_new.signatures['']
#print(infer(inputx=tf.constant([[10]], 'float'))['dense'].numpy())

#%%
model_new = tf.keras.models.load_model('keras_saved')
model_new.summary()

#%%
model_new = tf.saved_model.load('keras_saved')
model_new = tf.keras.models.Sequential([hub.KerasLayer(model_new,
                                                       input_shape=(1,),
                                                       output_shape=(1,))])
#instead of model_new, tensorflow_hub endpoint can be given,
#else module handler can be generated as hub.load(endpoint) --> can be passed

model_new.summary()
model_new.predict([10])

#%%
import PIL
image = PIL.Image.open('download.jpg').resize((160,160))
image = np.asarray(image)[None, :]
plt.imshow(image[0])

#%%
MODEL_SOURCE = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/classification/4'
model = hub.load(MODEL_SOURCE) #same as tf.saved_model.load(), called module handler
prediction = model(tf.constant(image/255.0, dtype='float')).numpy()
prediction = np.argmax(prediction)

#%%
model = tf.saved_model.load('mobilenet')
prediction = model(tf.constant(image/255.0, dtype='float')).numpy()
prediction = np.argmax(prediction)

#%% important dataset function
ds = tfds.load('mnist', as_supervised=True)
ds_numpy = tfds.as_numpy(ds)
iteration = ds_numpy['test']
temp = next(iteration)

#%%
tfds.load()