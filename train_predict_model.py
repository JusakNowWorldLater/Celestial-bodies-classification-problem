import pandas as pd
import numpy as np
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.optimizers import Adam

from keras import callbacks

from keras.layers.normalization import BatchNormalization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


batch_size    = 128

epochs        = 200

learning_rate = 0.0001

dropout       = 0.5

input_size     = 0
output_size    = 3

def load_data_from_csv(csv_path):
    global input_size
    db = pd.read_csv(csv_path)

    col_size = db.shape[1]

    if(input_size == 0):
        input_size = col_size - 2

    X = db.iloc[:, 1:(col_size - 1)]
    Y_raw = db.iloc[:, (col_size - 1):col_size]
    Y = []
    for index, row in Y_raw.iterrows():
        if  (row[0] == 0):
            Y.append(np.array([1,0,0]))
        elif(row[0] == 1):
            Y.append(np.array([0,1,0]))
        else:
            Y.append(np.array([0,0,1]))

    X = X.values
    X[X == 'na'] = 0
    X = X.astype(np.float)
    Y = np.asarray(Y)
    return X, Y


def create_model(): 
    model = Sequential()

    model.add(BatchNormalization())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
    return model



history = {
    'train-loss': [], 'train-accuracy': [],
    'test-loss': [], 'test-accuracy': [],
}



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