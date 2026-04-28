import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
    return dict

weight_1 = np.random.randn(3072, 512) * 0.01
bias_1 = np.zeros((1, 512))
weight_2 = np.random.rand(3072, 512) * 0.01
bias_2 = np.zeros((1, 512))

batch_1 = unpickle('data/cifar-10-batches-py/data_batch_1')
raw_image = batch_1[b'data'][0]
clean_image = raw_image.astype('float32') / 255.0

image_noise = np.random.normal(loc=0.0, scale=2.0, size = clean_image.shape)
noisy_image = clean_image + 0.2 * image_noise

