import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image

model = keras.models.load_model('FCN/')
img = tf.keras.utils.load_img('test.png', color_mode="grayscale", target_size=(28, 28))
img_arr = np.array(img)
img_arr = img_arr.astype("float32")/255
img_arr = np.expand_dims(img_arr, -1)
img_arr = np.reshape(img_arr, (1, 28, 28, 1))
prediction = model.predict(img_arr)
print(prediction)
prediction_class = np.argmax(prediction)
print("Predicted: ", prediction_class)