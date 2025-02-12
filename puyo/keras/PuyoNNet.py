import argparse
import os
import sys
sys.path.append('..')
from utils import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import *

hasTPU = False
tryTPU = False
if(tryTPU):
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])

        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        hasTPU = True
    except:
        pass

def relu_bn(inputs):
    relu1 = relu(inputs)
    bn = BatchNormalization()(relu1)
    return bn

def residual_block(x, filters, kernel_size=3):
    y = Conv2D(kernel_size=kernel_size,
               strides= (1),
               filters=filters,
               padding="same")(x)

    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    y = BatchNormalization()(y)
    out = Add()([x, y])
    out = relu(out)

    return out

def value_head(input):
    conv1 = Conv2D(kernel_size=1,
                strides=1,
                filters=1,
                padding="same")(input)

    bn1 = BatchNormalization()(conv1)
    bn1_relu = relu(bn1)

    flat = Flatten()(bn1_relu)

    dense1 = Dense(256)(flat)
    dn_relu = relu(dense1)

    dense2 = Dense(256)(dn_relu)

    return dense2

def policy_head(input):
    conv1 = Conv2D(kernel_size=2,
                strides=1,
                filters=1,
                padding="same")(input)
    bn1 = BatchNormalization()(conv1)
    bn1_relu = relu(bn1)
    flat = Flatten()(bn1_relu)
    return flat

class PuyoNNet():

    def createModel(self, game, args):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        # Inputs
        self.input_boards = Input(shape=(self.board_x, self.board_y))
        inputs = Reshape((self.board_x, self.board_y, 1))(self.input_boards)


        bn1 = BatchNormalization()(inputs)
        conv1 = Conv2D(args.num_channels, kernel_size=3, strides=1, padding="same")(bn1)
        t = relu_bn(conv1)

        for i in range(self.args.num_residual_layers):
            t = residual_block(t, filters=self.args.num_channels)

        self.pi = Dense(self.action_size, activation='softmax', name='pi')(policy_head(t))
        self.v = Dense(1, activation='tanh', name='v')(value_head(t))
    def __init__(self, game, args):
        # game params
        
        if(hasTPU):
            with strategy.scope():
                self.createModel(game, args)
                self.model = Model(inputs=self.input_boards,outputs=[self.pi, self.v])
                self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
        else:
            self.createModel(game, args)
            self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
            self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
