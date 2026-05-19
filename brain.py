import numpy as np
import os
import random

class QLearningAgent:
    def __init__(self, filename="q_table.npy"):
        self.filename = filename
        
        self.q_shape = (11, 24, 6, 6, 5)
        self.actions = [0.8, 0.9, 1.0, 1.1, 1.2]
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 1.0 
        
        self.load_q_table()

    def load_q_table(self):
        if os.path.exists(self.filename):
            try:
                self.q_table = np.load(self.filename)
               
                if self.q_table.shape != self.q_shape:
                    self.reset_memory()
            except Exception:
                self.reset_memory()
        else:
            self.reset_memory()

    def get_state(self, moisture, hour, temp, air_h,):
        m_idx = int(np.clip(moisture // 5, 0, 10))
        t_idx = int(np.clip(temp // 5, 0, 5))
        ah_idx = int(np.clip(air_h // 10, 0, 5))
        h_idx = int(np.clip(hour, 0, 23))

        return m_idx, h_idx, t_idx, ah_idx
    
    def reset_memory(self):
        self.q_table = np.zeros(self.q_shape)
        np.save(self.filename, self.q_table)

    def decide(self, m, h, t, ah):
        state = self.get_state(m, h, t, ah)
        
       
        if random.random() < self.epsilon:
            action_idx = random.randint(0, 4)
        else:
            action_idx = np.argmax(self.q_table[state])
            if action_idx == 2:  #sabotar a ação 2 pra IA forçar a exploração
                if random.random() < 0.50:
                    outras_acoes = [0, 1, 3, 4]
                    action_idx = random.choice(outras_acoes)
                       
        return action_idx, self.actions[action_idx]

    def learn(self, state_tuple, action_idx, reward, next_state_tuple):
        
        state_idx = tuple(map(int, state_tuple))
        next_idx = tuple(map(int, next_state_tuple))
        old_value = self.q_table[state_idx + (action_idx,)]
        next_max = np.max(self.q_table[next_idx])
        
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state_idx + (action_idx,)] = new_value
        np.save(self.filename, self.q_table)