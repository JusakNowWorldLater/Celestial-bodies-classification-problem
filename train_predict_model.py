import pandas as pd
import numpy as np
import os
import sklearn

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.optimizers import Adam

from keras import callbacks

from keras.layers.normalization import BatchNormalization

from sklearn.metrics import f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


batch_size    = 256

epochs        = 300

learning_rate = 0.0003

dropout       = 0.5

input_size     = 0
output_size    = 3

def load_data_from_csv(csv_path):
    db = pd.read_csv(csv_path)
    db = db.drop(columns=['objid', 'ra', 'dec','rowc','colc', 'rowv', 'colv'])
    col_size = db.shape[1]

    X = db.drop(columns=['class'])
    Y_raw = db['class']
    Y = []
    for val in Y_raw:
        if  (val == 0):
            Y.append(np.array([1,0,0]))
        elif(val == 1):
            Y.append(np.array([0,1,0]))
        else:
            Y.append(np.array([0,0,1]))

    X = X.values
    X[X == 'na'] = 0
    X = X.astype(np.float)
    Y = np.asarray(Y)
    return X, Y

def getIdxOfMax(arr):
    idx=0
    maxIdx = 0
    max=-1
    for i in arr:
        if(max < i):
            max = i
            maxIdx = idx
        idx += 1
    return maxIdx


def toOrigFormat(Y_val):
    Y_res = []
    for i in Y_val:
        Y_res.append(getIdxOfMax(i))
    return Y_res

def create_model(): 
    model = Sequential()

    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
    return model

X_train, Y_train = load_data_from_csv('data/train.csv')
X_test, Y_test   = load_data_from_csv('data/val.csv')

model = create_model()


history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    shuffle=True)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

Y_result = toOrigFormat(model.predict(X_test))
Y_test = toOrigFormat(Y_test)

print("Macro-averaged f1-score: ", f1_score(Y_test, Y_result, average='macro'))