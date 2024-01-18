#引入依赖
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#使数据集换个形状,加上通道数
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

#检查形状
print(X_train.shape)
print(X_test.shape)

#标准化像素值,X_train的每个值都/255
X_train = X_train/255
X_test = X_test/255

#定义模型
model = Sequential()

#加入卷积层
model.add(Conv2D(32,(3,3),activation = 'relu',input_shape=(28,28,1)))

#加入池化层
model.add(MaxPool2D(2,2))

#加入全连接层,relu函数用于将负数返回0
model.add(Flatten())
model.add(Dense(10,activation='relu'))

#加入输出层，softmax输出概率分布
#softmax将每个类别原始输出映射到（0，1）之间，保证所有概率和为1
model.add(Dense(10,activation='softmax'))

#编译模型
model.compile(loss='sparse_categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])

#拟合模型
model.fit(X_train,y_train,epochs =10)

#评估模型
model.evaluate(X_test,y_test)