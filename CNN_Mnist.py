#!/usr/bin/env python
# coding: utf-8

# ## CNN 卷積神經網路

# * 影像的特徵提取: 透過 Convolution 與 Max Pooling 提取影像特徵.
# * Fully connected Feedforward network: Flatten layers, hidden layers and output layers
# 
# http://puremonkey2010.blogspot.com/2017/07/toolkit-keras-mnist-cnn.html

# ## STEP1. 資料讀取與轉換 

# In[1]:


from keras.datasets import mnist  
from keras.utils import np_utils  
import numpy as np  
np.random.seed(10)  
(X_Train, y_Train), (X_Test, y_Test) = mnist.load_data()  
  
# Translation of data  
X_Train4D = X_Train.reshape(X_Train.shape[0], 28, 28, 1).astype('float32')  
X_Test4D = X_Test.reshape(X_Test.shape[0], 28, 28, 1).astype('float32') 


# ## STEP2. 將 Features 進行標準化與 Label 的 Onehot encoding 

# In[2]:


# Standardize feature data  
X_Train4D_norm = X_Train4D / 255  
X_Test4D_norm = X_Test4D /255  
  
# Label Onehot-encoding  
y_TrainOneHot = np_utils.to_categorical(y_Train)  
y_TestOneHot = np_utils.to_categorical(y_Test) 


# ## STEP3. 建立卷積層與池化層

# In[3]:


from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D  
  
model = Sequential()  
# Create CN layer 1  
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))  
# Create Max-Pool 1  
model.add(MaxPooling2D(pool_size=(2,2)))  
  
# Create CN layer 2  
model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))  
  
# Create Max-Pool 2  
model.add(MaxPooling2D(pool_size=(2,2)))


# ## STEP4. 建立神經網路 

# In[4]:


# Add Dropout layer  
model.add(Dropout(0.25))  
model.add(Flatten())  
model.add(Dense(128, activation='relu'))  
model.add(Dropout(0.5)) 
model.add(Dense(10, activation='softmax'))  
model.summary()  
print("")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# ## STEP5. 定義訓練並進行訓練 

# In[5]:


from keras import optimizers
from keras.callbacks import ModelCheckpoint 
import os
from datetime import datetime
from tensorflow import keras

# 定義訓練方式 
path='./model9527'
if not os.path.isdir(path):
 os.mkdir(path)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
logdir = path
logdir = logdir + "/logs/scalars/" 
logdir = logdir + datetime.now().strftime("%Y%m%d-%H%M%S")

#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
filepath=path

filepath = filepath+"/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
#mode='max')
#callbacks_list = [checkpoint]
callbacks_list =[ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max'),keras.callbacks.TensorBoard(log_dir=logdir,write_graph=False)]
                                  
                 


#train_history = model.fit(x=X_Train4D_norm,y=y_TrainOneHot, validation_split=0.2,epochs=10, batch_size=300, callbacks=callbacks_list,verbose=1) 
train_history = model.fit(x=X_Train4D_norm,y=y_TrainOneHot,epochs=15, batch_size=300,verbose=0,validation_data=(X_Test4D_norm, y_TestOneHot),callbacks=callbacks_list) 
# 開始訓練  
  
model.save( path+'/model.h5')


# ## STEP6. 畫出 accuracy 執行結果 

# In[36]:


import matplotlib.pyplot as plt  
import tensorflow as tf

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show() 
    
def plot_images_labels_predict(images, labels, prediction, idx, num=10):  
    fig = plt.gcf()  
    fig.set_size_inches(12, 14)  
    if num > 25: num = 25  
    for i in range(0, num):  
        ax=plt.subplot(5,5, 1+i)  
        ax.imshow(images[idx], cmap='binary')  
        title = "l=" + str(labels[idx])  
        if len(prediction) > 0:  
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))  
        else:  
            title = "l={}".format(str(labels[idx]))  
        ax.set_title(title, fontsize=10)  
        ax.set_xticks([]); ax.set_yticks([])  
        idx+=1  
plt.show()  

show_train_history(train_history, 'acc', 'val_acc')  
show_train_history(train_history, 'loss', 'val_loss') 

scores = model.evaluate(X_Test4D_norm, y_TestOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))   

print("\t[Info] Making prediction of X_Test4D_norm")  
prediction = model.predict_classes(X_Test4D_norm)  # Making prediction and save result to prediction  
print()  
print("\t[Info] Show 10 prediction result (From 240):")  
print("%s\n" % (prediction[250:260]))
plot_images_labels_predict(X_Test, y_Test, prediction, idx=250)  
import pandas as pd
print("\t[Info] Display Confusion Matrix:")
print("%s\n" % pd.crosstab(y_Test, prediction, rownames=['label'], colnames=['predict']))


# ## STEP7. 匯出模型

# In[37]:


model.save_weights('/tmp/CNN_Mnist.h5')


# ## STEP8. 讀取模型

# In[38]:


from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D 

model.load_weights('/tmp/CNN_Mnist.h5')


# ## STEP9. 測試模型

# In[39]:


import matplotlib.pyplot as plt  
import tensorflow as tf

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show() 
    
def plot_images_labels_predict(images, labels, prediction, idx, num=10):  
    fig = plt.gcf()  
    fig.set_size_inches(12, 14)  
    if num > 25: num = 25  
    for i in range(0, num):  
        ax=plt.subplot(5,5, 1+i)  
        ax.imshow(images[idx], cmap='binary')  
        title = "l=" + str(labels[idx])  
        if len(prediction) > 0:  
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))  
        else:  
            title = "l={}".format(str(labels[idx]))  
        ax.set_title(title, fontsize=10)  
        ax.set_xticks([]); ax.set_yticks([])  
        idx+=1  
plt.show()  

scores = model.evaluate(X_Test4D_norm, y_TestOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))   

print("\t[Info] Making prediction of X_Test4D_norm")  
prediction = model.predict_classes(X_Test4D_norm)  # Making prediction and save result to prediction  
print()  
print("\t[Info] Show 10 prediction result (From 240):")  
print("%s\n" % (prediction[250:260]))
plot_images_labels_predict(X_Test, y_Test, prediction, idx=250)  
import pandas as pd
print("\t[Info] Display Confusion Matrix:")
print("%s\n" % pd.crosstab(y_Test, prediction, rownames=['label'], colnames=['predict']))


# In[ ]:





# In[ ]:




