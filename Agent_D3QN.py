import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from ReplayBuffer_D3QN import ReplayBuffer
from DuelingDDQN import D3QN

class D3QNAgent():

    def __init__(self,  lr, gamma, n_actions,epsilon, batch_size, input_shape,
                 eps_decay=1e-3, eps_end=0.01, mem_size=1000000, layer1_size=256,
                 layer2_size=256, replace=100):

        self.action_space=[i for i in range(n_actions)]
        self.lr=lr
        self.gamma=gamma
        self.n_actions=n_actions
        self.epsilon=epsilon
        self.batch_size=batch_size
        self.input_shape=input_shape
        self.eps_decay=eps_decay
        self.eps_end=eps_end
        self.mem_size=mem_size
        self.layer1_size=layer1_size
        self.layer2_size=layer2_size
        self.replace=replace
        self.learn_step_counter=0

        self.memory = ReplayBuffer(self.mem_size, self.input_shape, self.n_actions)
        self.q_eval = D3QN(self.n_actions, self.layer1_size, self.layer2_size)
        self.q_next = D3QN(self.n_actions, self.layer1_size, self.layer2_size)  # Q target

        self.q_eval.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
        self.q_next.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    def store_transition(self, state, newstate, action, reward, done):
        self.memory.store_transition(state,newstate,action,reward,done)

    def choose_action(self, observation):
        actions_total = []
        state = np.array([observation])
        if np.random.random() < self.epsilon:
            actions_total = random.sample(range(self.n_actions), self.n_actions)
        else:
            for i in range(self.n_actions):
                actions = self.q_eval.advantage(state)
                action = tf.math.argmax(actions, axis=1).numpy()[0]
                actions_total.append(action)
        return actions_total

    def learn(self):
        if self.learn_step_counter < self.batch_size:
            return
        if self.learn_step_counter % self.replace ==0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, newstates, actions, rewards, dones= self.memory.sample_transition(self.batch_size)

        q_pred=self.q_eval(states)
        q_next=self.q_next(newstates)
        q_target=q_pred.numpy()  #  y
        max_action=tf.math.argmax(self.q_eval(newstates),axis=1)

        for idx, done in enumerate(dones):
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma * (
                        1 - int(dones[idx]) * q_next[idx, max_action[idx]])

        self.q_eval.train_on_batch(states, q_target)
        if self.epsilon > self.eps_end:
            self.epsilon=self.epsilon-self.eps_decay
        else:
            self.epsilon=self.eps_end

        self.learn_step_counter+=1







