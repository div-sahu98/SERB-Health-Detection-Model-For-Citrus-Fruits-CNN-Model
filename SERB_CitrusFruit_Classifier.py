#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[22]:


import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pickle
import cv2


# In[23]:


DIRECTORY=r'C:\Users\Divyansh\Desktop\computer-vision'
CATEGORIES=['freshoranges','rottenoranges']


# In[24]:


IMG_SIZE=100

data= []

for category in CATEGORIES:
    folder=os.path.join(DIRECTORY,category)
    label= CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path =os.path.join(folder,img)
        img_arr=cv2.imread(img_path)
        img_arr= cv2.resize(img_arr, (100,100)) 
        data.append([img_arr,label])
        


# In[ ]:





# In[25]:


random.shuffle(data)


# In[26]:


X=[]
y=[]
for features,labels in data:
    X.append(features)
    y.append(labels)


# In[27]:


X =np.array(X)
y=np.array(y)


# In[ ]:





# In[28]:


X=X/255


# In[ ]:





# In[ ]:





# In[11]:


from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.callbacks import TensorBoard
import time


# In[30]:


NAME=f'orange-pridiction-{int(time.time())}'
tensorboard=TensorBoard(log_dir=r'C:\Users\Divyansh\Desktop\computer-vision')

model = Sequential()

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,input_shape=X.shape[1:],activation='relu'))

model.add(Dense(128,activation='relu'))

model.add(Dense(2,activation='softmax'))


# In[31]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[32]:


model.fit(X,y,epochs=5,validation_split=0.2,callbacks=[tensorboard])


# In[33]:


model.evaluate(X,y)


# In[ ]:





# In[34]:


X.shape


# In[39]:


DIRECTORY=r'C:\Users\Divyansh\Desktop\check'
CATEGORIES=['freshoranges','rottenoranges']
folder=os.path.join(DIRECTORY,category)
testing=[]
for category in CATEGORIES:
    folder=os.path.join(DIRECTORY,category)
    label= CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path =os.path.join(folder,img)
        img_arr=cv2.imread(img_path)
        img_arr= cv2.resize(img_arr, (100,100)) 
        testing.append([img_arr,label])
X_test=[]
y_test=[]
for features,labels in testing:
    X_test.append(features)
    y_test.append(labels)


# In[44]:


X_test =np.array(X_test)
y_test=np.array(y_test)

X_test=X_test/255
y_test=y_test/255


# In[45]:


X_test.shape


# In[46]:


y_test.shape


# In[47]:


y_pred = model.predict(X_test)
y_pred[:5]


# In[48]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[50]:


plt.imshow(X_test[5])
plt.xlabel([y_test[5]])


# In[ ]:




