import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_mem, input_shape, n_actions):
        self.memorySize=max_mem
        self.couter=0

        self.state_transition = np.zeros((self.memorySize, *input_shape), dtype=np.float32)
        self.newstate_transition = np.zeros((self.memorySize, *input_shape), dtype=np.float32)
        self.action_transition=np.zeros((self.memorySize, n_actions), dtype=np.int32)
        self.reward_transition = np.zeros(self.memorySize, dtype=np.float32)
        self.terminal_transition=np.zeros(self.memorySize, dtype=np.bool)

    def store_transition(self, state, newstate, action, reward, done):
        index=self.couter % self.memorySize

        self.state_transition[index]=state
        self.newstate_transition[index]=newstate
        self.action_transition[index] = action
        self.reward_transition[index] = reward
        self.terminal_transition[index] = done

        self.couter+=1

    def sample_transition(self, batch_size):
        max_mem=min(self.couter, self.memorySize)
        batch=np.random.choice(max_mem, batch_size, replace=False)

        state=self.state_transition[batch]
        newstate=self.newstate_transition[batch]
        action=self.action_transition[batch]
        reward=self.reward_transition[batch]
        done=self.terminal_transition[batch]

        return state, newstate, action, reward, done



