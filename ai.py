import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Flatten())

# #relu: (f(x)=x if x>0 and f(x)=0 if x<=0)
# model.add(tf.keras.layers.Dense(128,activation='relu'))

# model.add(tf.keras.layers.Dense(128,activation='relu'))
# #softmax: vector of K real numbers into a probability distribution of K possible outcomes. k=10 in this case
# model.add(tf.keras.layers.Dense(10,activation='softmax'))

# model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# model.fit(x_train,y_train,epochs=5)

# model.save("digitRecon.h5")


# calling the model and evalutation the testing data and predicting with sample data
###################################################################
digitRecon = tf.keras.models.load_model("arabicocr.h5")

loss, accuracy = digitRecon.evaluate(x_test, y_test)
print(loss)
print(accuracy)

# img = cv2.imread(f"Images/b_arabic.jpg")[:, :, 0]
# img = np.invert(np.array([img]))
# plt.imshow(img[0],cmap=plt.cm.binary)
# plt.show()
# prediction = digitRecon.predict(img)

# print(f"prediction is : {np.argmax(prediction)}")
