import torch.optim as optim

# Edit from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, current_score):

        if self.best_score is None:
            self.best_score = current_score

        elif self.best_score < current_score + self.delta: # there is no improvement
            self.counter += 1
            # log("best {:.2f}, current {:.2f}".format(self.best_score, current_score))
            # log(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0

def get_optimizer(name):
    if name.lower() == 'rmsprop':
        return optim.RMSprop
    elif name.lower() == 'sgd':
        return optim.SGD
    else:
        raise NotImplementedError