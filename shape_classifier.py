import numpy as np
import cv2 as cv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split as tts


def show(a):
    cv.imshow('show', a)
    if cv.waitKey(1000) | 0xFF == ord("q"):
        cv.destroyAllWindows()


def load_defaults():
    classes = np.load('F:\Coding Projects\Visual Studio PYTHON\Shape_Classification_DL\datesets\Brown\Brown_classes_RGB.npy')
    labels = np.load('F:\Coding Projects\Visual Studio PYTHON\Shape_Classification_DL\datesets\Brown\Brown_labels_RGB.npy')
    total = np.load('F:\Coding Projects\Visual Studio PYTHON\Shape_Classification_DL\datesets\Brown\Brown_dataset_RGB.npy', allow_pickle=True)
    return classes, labels, total


classes, labels, total = load_defaults()

x = total
y = labels

# Split training and testing inputs
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0)

# Scaling input
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(len(classes), 'Number of Classes')

# Define dimenstions
num_classes = len(classes)
img_rows = 128
img_cols = 128
batch_size = 10
epochs = 20
input_shape = (img_rows, img_cols, 3)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

Loss = []
Accuracy = []
Epoch = []
for epochs in range(1, epochs, 3):
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    Loss.append(score[0])
    Accuracy.append(score[1])
    Epoch.append(epochs)

np.save('F:\Coding Projects\Visual Studio PYTHON\Shape_Classification_DL\datesets\Brown\Loss', Loss)
np.save('F:\Coding Projects\Visual Studio PYTHON\Shape_Classification_DL\datesets\Bown\Accuracy', Accuracy)
np.save('F:\Coding Projects\Visual Studio PYTHON\Shape_Classification_DL\datesets\Brown\Epoch', Epoch)
