import numpy as np


class exponential_beta_scheduler:
    def __init__(self, beta_start, beta_end, time_constant=300, num_epochs=300):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_constant = time_constant
        self.num_epochs = num_epochs
    def __call__(self, epoch):
        return self.beta_start + (self.beta_end - self.beta_start) * (1 - np.exp(-epoch / self.time_constant))/(1-np.exp(-self.num_epochs/self.time_constant))

class linear_beta_scheduler:
    def __init__(self, beta_start, beta_end, num_epochs=300):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_epochs = num_epochs
    def __call__(self, epoch):
        return self.beta_start + (self.beta_end - self.beta_start) * epoch/self.num_epochs