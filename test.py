import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def proc_img(imgs):
    return imgs.reshape(3, 32, 32).transpose(1, 2, 0)

batch1 = unpickle('cifar-10-batches-py/data_batch_1')
raw_img = batch1[b'data'][0]

clean_img = raw_img.astype('float32') / 255.0
noise_factor = 0.2
noise = np.random.normal(loc=0.0, scale=1.0, size=clean_img.shape)
noisy_img = clean_img + noise_factor * noise

noisy_img = np.clip(noisy_img, 0., 1.)

reshaped_img = raw_img.reshape(3, 32, 32)

final_img = reshaped_img.transpose(1, 2, 0)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(proc_img(clean_img))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(proc_img(noisy_img))
plt.title("Noisy Image")
plt.show()
