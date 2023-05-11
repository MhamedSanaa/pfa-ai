import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

x_train = pd.read_csv("archive/csvTrainImages 13440x1024.csv",header=None)
y_train = pd.read_csv("archive/csvTrainLabel 13440x1.csv",header=None)

x_test = pd.read_csv("archive/csvTestImages 3360x1024.csv",header=None)
y_test = pd.read_csv("archive/csvTestLabel 3360x1.csv",header=None)

print("x_train.shape =", x_train.shape, "\ny_train.shape =", y_train.shape, "\nx_test.shape =", x_test.shape, "\ny_test.shape =", y_test.shape)


x_train = x_train.iloc[:,:].values
x_test = x_test.iloc[:,:].values
y_train = y_train.iloc[:,:].values
y_test = y_test.iloc[:,:].values


x_train = x_train.reshape(-1,32,32,1)
x_test = x_test.reshape(-1,32,32,1)

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape, x_test.shape)


total_classes = len(np.unique(y_train))+1
print(total_classes)
y_train = tf.keras.utils.to_categorical(y_train,total_classes)
y_test = tf.keras.utils.to_categorical(y_test, total_classes)

y_train.shape


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(32,32,1)),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(total_classes, activation='softmax')
# ])
# model.summary()


#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(x_train,y_train,batch_size=128,epochs=80)


#model.save("arabicocr.h5")


digitRecon = tf.keras.models.load_model("arabicocr.h5")

loss, accuracy = digitRecon.evaluate(x_test, y_test)
print(loss)
print(accuracy)