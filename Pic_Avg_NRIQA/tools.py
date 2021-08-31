import numpy as np

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

