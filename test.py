import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def proc_img(imgs):
    return imgs.reshape(3, 32, 32).transpose(1, 2, 0)

def calc_psnr(target, pred):
    mse = np.mean((target - pred)**2)
    if mse == 0:
        return 100
    max_px = 1.0
    psnr = 20 * np.log10(max_px / np.sqrt(mse))
    return psnr


in_dimension = 3072
hid_dimension = 512

w_1 = np.random.randn(in_dimension, hid_dimension) * 0.01
b_1 = np.zeros((1, hid_dimension))
w_2 = np.random.randn(hid_dimension, in_dimension) * 0.01
b_2 = np.zeros((1, in_dimension))

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

init_psnr = calc_psnr(clean_img, noisy_img)
print(f"Starting PSNR: {init_psnr:.2f} dB")

res_noisy_img = noisy_img.reshape(1, 3072)
res_clean_img = clean_img.reshape(1, 3072)

for i in range(200):
    Z1 = np.dot(res_noisy_img, w_1) + b_1
    A1 = np.maximum(0, Z1)

    Z2 = np.dot(A1, w_2) + b_2
    fixed = 1 / (1 + np.exp(-Z2))

    dz2 = (fixed - clean_img.reshape(1, 3072)) * (fixed * (1 - fixed))
    dw2 = np.dot(A1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = np.dot(dz2, w_2.T)
    dz1 = da1 * (Z1 > 0)
    dw1 = np.dot(res_noisy_img.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    learning_rate = 0.1
    w_1 -= learning_rate * dw1
    b_1 -= learning_rate * db1
    w_2 -= learning_rate * dw2
    b_2 -= learning_rate * db2
    if i % 10 == 0:
        current_psnr = calc_psnr(clean_img.reshape(1, 3072), fixed)
        print(f"Iteration {i} PSNR: {current_psnr:.2f} dB")

plt.figure(figsize=(5, 5))
plt.imshow(proc_img(fixed))
plt.title(f"Reconstructed Image (PSNR: {calc_psnr(res_clean_img, fixed):.2f})")
plt.show()