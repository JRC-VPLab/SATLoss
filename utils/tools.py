import torch

def adj_lr(lr, epoch, decay, rate):
    # decay: list of decay epochs;
    r = 1.0
    decay = sorted(decay)
    for i in range(len(decay)):
        if epoch > decay[i]:
            r *= rate

    return lr * r
