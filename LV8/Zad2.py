import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import keras.models
from keras.models import load_model

num_classes = 10
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

model=load_model('FCN/')
model.summary()

predictions = model.predict(x_test_s)
score = model.evaluate(x_test_s,y_test_s,verbose=0)

predictions=np.argmax(predictions, axis=1) 
y_test_s=np.argmax(y_test_s, axis=1)
mistakes=x_test_s[y_test_s!=predictions]
mistakes_index=np.where(predictions!=y_test_s)

print(len(mistakes))
print("Predictions:", predictions[:10])
print("Actual values:", y_test_s[:10])

for i in range(len(mistakes_index[0])):
    index = mistakes_index[0][i]
    plt.figure()
    plt.imshow(mistakes[i])
    plt.title("Stvarna klasa: " + str(y_test_s[index]) + " Predikcija: " + str(predictions[index]))
    plt.show()