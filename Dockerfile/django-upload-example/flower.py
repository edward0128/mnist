import tensorflow as tf
from PIL import Image
from PIL import Image
import numpy as np
import tflearn.datasets.mnist as mnist
#from keras.datasets import mnist  
from keras.utils import np_utils
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D 

import os
import string
import sys


from keras.datasets import mnist  
from keras.utils import np_utils  
import numpy as np  





im = Image.open(sys.argv[1])
im = im.convert("L")
im = im.resize((28,28))  # 更改图像宽和高
im.save('/mnt/inference/1234.png')
img = Image.open("/mnt/inference/1234.png")
arr = np.array(img)
(X_Train, y_Train), (X_Test, y_Test) = mnist.load_data()
X_Test[0]=arr;
X_Train4D = X_Train.reshape(X_Train.shape[0], 28, 28, 1).astype('float32')  
X_Test4D = X_Test.reshape(X_Test.shape[0], 28, 28, 1).astype('float32')
# Standardize feature data  
X_Train4D_norm = X_Train4D / 255  
X_Test4D_norm = X_Test4D /255  
  
# Label Onehot-encoding  
y_TrainOneHot = np_utils.to_categorical(y_Train)  
y_TestOneHot = np_utils.to_categorical(y_Test)


try:
    model
except NameError:
    var_exists = False
else:
    var_exists = True

if var_exists == False:
    print("result:")
    model = tf.contrib.keras.models.load_model('/mnt/CNN_Mnist.h5')


#import matplotlib.pyplot as plt  
#import tensorflow as tf

prediction = model.predict_classes(X_Test4D_norm[0:1])  
 
print("%s\n" % (prediction[0:1]))



#os.chdir('/mnt/student1/tensorflow-for-poets-2/')

#test="python -m scripts.inference --graph=tf_files/retrained_graph.pb --image=";
#test=test+sys.argv[1]
#os.system(test)
