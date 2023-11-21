#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install tensorflow')
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[5]:


(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()


# In[7]:


len(x_train)


# In[8]:


len(x_test)


# In[9]:


x_train[1].shape


# In[10]:


x_train[1]


# In[11]:


plt.matshow(x_train[1])


# In[12]:


y_train[1]


# In[13]:


x_train=x_train/255
x_test=x_test/255


# In[14]:


x_train[1]


# In[16]:


x_train_flattened=x_train.reshape(len(x_train),28*28)
x_test_flattened=x_test.reshape(len(x_test),28*28)


# In[17]:


x_train_flattened.shape


# In[18]:


x_train_flattened[1]


# In[22]:


model=keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,), activation='sigmoid')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_flattened,y_train,epochs=5)


# In[23]:


model.evaluate(x_test_flattened,y_test)


# In[24]:


y_predicted=model.predict(x_test_flattened)
y_predicted[1]


# In[27]:


plt.matshow(x_test[1])


# In[28]:


np.argmax(y_predicted[1])


# In[30]:


y_predicted_labels=[np.argmax(i) for i in y_predicted]


# In[32]:


y_predicted_labels[:10]


# In[33]:


c_matrix= tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(c_matrix,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[40]:


model = keras.Sequential([
    keras.layers.Dense(128,input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_flattened,y_train,epochs=10)


# In[39]:


model.evaluate(x_test_flattened,y_test)


# In[42]:


y_predicted=model.predict(x_test_flattened)
y_predicted_labels=[np.argmax(i) for i in y_predicted]
c_matrix=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize=(10,7))
sn.heatmap(c_matrix,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')



# In[45]:


model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10)


# In[ ]:


model.evaluate(x_test,y_test)

