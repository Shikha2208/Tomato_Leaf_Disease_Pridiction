#!/usr/bin/env python
# coding: utf-8

# ## Transfer Learning Inception V3 using Keras

# Please download the dataset from the below url

# In[1]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[2]:


# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt


# In[3]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Datasets/train'
valid_path = 'Datasets/test'


# In[4]:


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



# In[5]:


# don't train existing weights
for layer in inception.layers:
    layer.trainable = False


# In[6]:


# useful for getting number of output classes
folders = glob('Datasets/train/*')


# In[7]:


# our layers - you can add more if you want
x = Flatten()(inception.output)


# In[8]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=inception.input, outputs=prediction)


# In[ ]:



# view the structure of the model
model.summary()


# In[ ]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[ ]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[ ]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[ ]:


test_set = test_datagen.flow_from_directory('Datasets/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[ ]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[ ]:


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_inception.h5')


# In[ ]:





# In[ ]:



y_pred = model.predict(test_set)


# In[ ]:


y_pred


# In[ ]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[ ]:


y_pred


# In[ ]:





# In[ ]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[ ]:


model=load_model('model_resnet50.h5')


# In[ ]:


img_data


# In[ ]:


img=image.load_img('Datasets/Test/Coffee/download (2).jpg',target_size=(224,224))


# In[ ]:


x=image.img_to_array(img)
x


# In[ ]:


x.shape


# In[ ]:


x=x/255


# In[ ]:


import numpy as np
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape


# In[ ]:


model.predict(img_data)


# In[ ]:


a=np.argmax(model.predict(img_data), axis=1)


# In[ ]:


a==1


# In[ ]:


import tensorflow as tf


# In[ ]:


tf.__version__


# In[ ]:




