import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import numpy as np
import cv2 as cv
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

import pickle
# --------------- Preprocessing Database -----------------------------



def extract_label():
    complete_text = open(r"C:\Users\Avinash\Desktop\bccn\all-mias_info.txt")
    lines = [line.rstrip('\n') for line in complete_text]

    details = {}
    count = 0
    for line in lines:
        value = line.split(' ')

        if len(value) == 4:
            details[value[0]] = 0
        else:
            if value[3] == 'B':
                details[value[0]] = 1
            if value[3] == 'M':
                details[value[0]] = 2
    return details


def extract_image():
    details = {}

    current_path = os.getcwd()
    for i in range(322):
        path = os.path.join(current_path, 'all-mias')
        if i < 9:
            path = path + '\\mdb00' + str(i + 1) + ".pgm"
            filelabel = 'mdb00' + str(i + 1)
        elif i < 99:
            path = path + '\\mdb0' + str(i + 1) + ".pgm"
            filelabel = 'mdb0' + str(i + 1)
        else:
            path = path + '\\mdb' + str(i + 1) + ".pgm"
            filelabel = 'mdb' + str(i + 1)

        img = 0

        # dimensions = (64,64)
        try:
            img = Image.open(path)
            img = img.resize((64, 64))
            img = np.array(img)
            info = np.iinfo(img.dtype)
            # img = img.astype(np.uint8) / info.max  # normalize the data to 0 - 1
            # img = 255 * img  # Now scale by 255
            img = img.astype(np.float32)
        except Exception as e:
            print(path)
            print(str(e))

        details[filelabel] = img

    return details


# --------------- Creating the test, train split ----------------------

def spl():
    labels = extract_label()
    images = extract_image()
    imageids = labels.keys()

    X = []
    Y = []

    for id in imageids:
        X.append(images[id])
        Y.append(labels[id])

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

    a, b, c = X_train.shape

    X_train = np.reshape(X_train, (a, b, c, 1))  # 1 for gray scale
    a, b, c = X_test.shape

    X_test = np.reshape(X_test, (a, b, c, 1))  # 1 for gray scale
    return X_train, Y_train, X_test, Y_test


# ----------------- CNN Model -----------------------------------------


# ------------------- Main function ------------------------------

X_train, y_train, X_test, y_test = spl()
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)
model = Sequential()  # 64x64
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))  # 62x62
model.add(MaxPool2D(pool_size=(2, 2)))  # 31X31

model.add(Conv2D(64, (3, 3), activation='relu'))  # 29X29
model.add(MaxPool2D(pool_size=(2, 2)))  # 15x15

model.add(Conv2D(64, (3, 3), activation='relu'))  # 13x13
model.add(MaxPool2D(pool_size=(2, 2)))  # 7x7

model.add(Flatten())

model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
# loss, accuracy = model.evaluate(X_train, y_train)
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print("Test: accuracy = %f  ;  loss = %f" % (100 * accuracy, loss))

# pred = model.predict(X_test)
# print(accuracy_score(y_test, pred))
# probabilities = model.predict(X_test)
# predictions = [float(np.round(x)) for x in probabilities]
# accuracy1 = np.mean(predictions == y_test)
# print("Prediction Accuracy: %.2f%%" % (accuracy1 * 100))

#model.save('result.h5')
with open('bcc_cnn_ps1.pkl', 'wb') as file:
    pickle.dump(model, file)