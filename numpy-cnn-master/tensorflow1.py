import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# 加载和预处理 CIFAR-10 数据集
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
#展示数据集样式
plt.imshow(train_images[1])
plt.title(f'CIFAR-10 Digit: {train_labels[1]}')
plt.show()
# # 使用 tf.data.Dataset 创建数据集
# batch_size = 64
# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(batch_size)
# test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
#
# # 创建一个简单的 CNN 模型
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
# # 编译模型
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# # 训练模型
# epochs = 10
# history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)
#
# # 评估模型
# test_loss, test_acc = model.evaluate(test_dataset)
# print(f'Test accuracy: {test_acc}')
#
# # 绘制训练历史
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
#
# #绘制性能图（损失图）
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss Over Epochs')
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()