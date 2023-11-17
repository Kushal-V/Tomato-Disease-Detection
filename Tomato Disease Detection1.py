#!/usr/bin/env python
# coding: utf-8

# In[62]:


import os
import tensorflow 
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing import image


# In[64]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)


# In[65]:


batch_size = 32

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\ASUS\Downloads\Tomato Plant\tomato\train',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    r'C:\Users\ASUS\Downloads\Tomato Plant\tomato\val',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)


# In[66]:


model = Sequential()

model.add(Conv2D(64, (3, 3), strides=(2, 2), input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), strides=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(len(train_generator.class_indices), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[67]:


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10, 
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)


# In[68]:


accuracy = model.evaluate(val_generator, steps=val_generator.samples // batch_size)
print(f'Validation Accuracy: {accuracy[1]*100:.2f}%')


# In[83]:


img_path = r"C:\Users\ASUS\Downloads\tomato input12.jpeg"
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0 

prediction = model.predict(img_array)

class_indices = train_generator.class_indices
predicted_class = list(class_indices.keys())[np.argmax(prediction)]
print(f'The detected disease is: {predicted_class}')

