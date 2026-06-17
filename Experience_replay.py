from collections import deque
import random

class Replay_memory():
    # create FIFO
    def __init__(self, maxlen, seed=None): # max length shows the number of lenght of epereicen memory means how many experiences can be stored there
        self.momory = deque([], maxlen=maxlen)
    
    def append(self,new_exp):
        self.momory.append(new_exp)
    
    def sample(self, sample_size):
        return random.sample(self.momory, sample_size)
    
    # curr buffer size
    def __len__(self):
        return len(self.momory)