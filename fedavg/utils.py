"""
Based on the implementation in https://github.com/AshwinRJ/Federated-Learning-PyTorch
"""

import copy
import torch


def average_weights(w, counts):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = 0
        for i in range(len(w)):
            w_avg[key] += torch.mul(w[i][key], counts[i]/sum(counts))
        
    return w_avg
