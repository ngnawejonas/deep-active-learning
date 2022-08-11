import numpy as np
import torch.optim as optim

def get_optimizer(name):
    if name.lower() == 'rmsprop':
        return optim.RMSprop
    elif name.lower() == 'sgd':
        return optim.SGD
    elif name.lower() == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError

def log_to_file(file_name, line):
    filepath = 'results/'+file_name
    file = open(filepath, 'a')
    file.write(line)
    if not line.endswith('\n'):
        file.write('\n')
    file.close()

