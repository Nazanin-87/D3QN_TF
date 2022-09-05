import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np

class D3QN(keras.Model):

    def __init__(self, n_actions, layer1_dim, layer2_dim):
        super(D3QN, self).__init__()
        self.L1 = keras.layers.Dense(layer1_dim, activation='relu')
        self.L2 = keras.layers.Dense(layer2_dim, activation='relu')
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self,state):
        x = self.L1(state)
        x = self.L2(x)
        V = self.V(x)
        A = self.A(x)
        Q = V + (A - tf.math.reduce_mean(A, axis=1, keepdim=True))
        return Q

    def advantage(self, state):
        x = self.L1(state)
        x = self.L2(x)
        A = self.A(x)
        return A





