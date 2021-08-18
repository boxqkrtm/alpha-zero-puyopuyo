import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
from Duel import *
import time
import os
import matplotlib.pyplot as plt
import gc

# Need "chcp 65001" for print
os.system("chcp 65001")


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(
    16, (3, 3), activation='relu', input_shape=(14, 14, 1)))
model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='SGD',
              loss='mean_squared_error',
              metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=5)
model.load_weights("./model")


def modelPredict(d, playerNum, rand=0):
    global model
    # if random move
    if(random.random() <= rand):
        return random.choice(d.GetValidMovesPlayer(playerNum))
    valid = d.GetValidMovesPlayer(playerNum)

    data = []
    dd = Duel(seed=-1)
    for i in valid:
        movedD = Duel(duel=d)
        if(playerNum == 0):
            movedD.input(i, -1)
        else:
            movedD.input(-1, i)
        movedD.run()
        data += [dd.GrayScaleArray(movedD.getGameInfo(playerNum))]
        del movedD
    del dd
    data = np.array(data)
    return valid[np.argmax(model.predict(data))]


def trainAndSave(data, datan):
    model.fit(data, datan, epochs=5)
    model.save_weights("./model")


def RandomModelPlay(rand=0.9, view=False):
    data = []
    datan = []
    d = Duel()
    for i in range(1000):
        d.reset()
        state = 2
        ai1, ai2 = 0, 0
        drr = []
        if(view == True):
            h = plt.imshow(d.GrayScaleArray().reshape(14, 14))
        while(True):
            if(state == 0 or state == 2):
                drr += [Duel(duel=d)]
                ai1 = modelPredict(d, 0, rand)
            if(state == 1 or state == 2):
                drr += [Duel(duel=d)]
                ai2 = modelPredict(d, 1, rand)
            d.input(ai1, ai2)
            state = d.run()
            if(view == True):
                d.print()
                h.set_data(d.GrayScaleArray().reshape(14, 14))
                plt.draw(), plt.pause(0.04)
            if(state >= 3):
                drr += [Duel(duel=d)]
                break

        if(state == 5):
            continue
        # print(d.GrayScaleArray().reshape(14, 14))

        for j in drr:
            data += [j.GrayScaleArray()]
            datan += [1] if state == 3 else [-1]
        # gc.collect()
        #h.set_data(d.GrayScaleArray().reshape(14, 14))
        print("시뮬레이션중", i, "/", 1000, cnt)

    # train
    data = np.array(data)
    datan = np.array(datan)
    trainAndSave(data, datan)


rand = 0.99
cnt = 6
while(True):
    RandomModelPlay(0, True)
    cnt += 1
