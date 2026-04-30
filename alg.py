import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
    return dict

weight_1 = np.random.randn(3072, 1024) * np.sqrt(1. /3072)
bias_1 = np.zeros((1, 1024))
weight_2 = np.random.randn(1024, 3072) * np.sqrt(1./1024)
bias_2 = np.zeros((1, 3072))

batch_1 = unpickle('data/cifar-10-batches-py/data_batch_1')
raw_image = batch_1[b'data'][:2000]
clean_image = raw_image.astype('float32') / 255.0

image_noise = np.random.normal(loc=0.0, scale=1.0, size = clean_image.shape)
noisy_image = clean_image + 0.2 * image_noise
noisy_image = np.clip(noisy_image, 0, 1)

weight_class = np.random.randn(1024, 10) * np.sqrt(1./1024)
bias_class= np.zeros((1, 10))

decay = 0.0005
lr = 0.001
batch_size=128
num_samples = 1000
labels = np.array(batch_1[b'labels'][:2000])
for e in range(500):
    indices = np.random.permutation(num_samples)
    noisy_sfl = noisy_image[indices]
    clean_sfl = clean_image[indices]
    labels_sfl = labels[indices]
    

    for i in range(0, num_samples, batch_size):
        batch_x = noisy_sfl[i: i + batch_size]
        batch_y = clean_sfl[i:i + batch_size]
        l_batch = labels_sfl[i:i+batch_size]
        current_length = len(l_batch)
                             
        layer_1=np.dot(batch_x, weight_1) + bias_1

        relu = np.where(layer_1 > 0, layer_1, layer_1 * 0.01) 
        layer_2=np.dot(relu, weight_2) + bias_2
        sigmoid= 1 / (1 + np.exp(-layer_2))      

                       
    
        class_val = np.dot(relu, weight_class) + bias_class
        expected_val = np.exp(class_val - np.max(class_val, axis=1, keepdims=True))
        probabilities = expected_val / np.sum(expected_val, axis=1, keepdims=True)

        error_layer_2 = (sigmoid - batch_y) * (sigmoid * (1-sigmoid))
        grad_score = probabilities.copy()
        grad_score[range(current_length), l_batch] -= 1
        grad_score /= current_length
        grad_relu = np.where(layer_1 > 0, 1, 0.01)

        error_layer_1 = (np.dot(error_layer_2, weight_2.T) +  0.1 * np.dot(grad_score, weight_class.T))* grad_relu
        gradient_weight_2 = np.dot(relu.T, error_layer_2) / current_length + decay * weight_2
        gradient_weight_1 = np.dot(batch_x.T, error_layer_1) / current_length + decay * weight_1
        grad_weight_class = np.dot(relu.T, grad_score) + decay * weight_class
        gradient_bias_2 = np.sum(error_layer_2, axis=0, keepdims=True) / current_length
        gradient_bias_1 = np.sum(error_layer_1, axis=0, keepdims=True) / current_length
        grad_bias_class = np.sum(grad_score, axis=0, keepdims=True)



        weight_1 = weight_1 - (lr * gradient_weight_1)
        weight_2 = weight_2 - (lr * gradient_weight_2)
        bias_1 = bias_1 - (lr * gradient_bias_1)
        bias_2 = bias_2 - (lr * gradient_bias_2)

        weight_class -= lr * grad_weight_class
        bias_class -= lr * grad_bias_class

    if e % 10 == 0:
        if e > 0 and e % 100 == 0:
            lr *= 0.5

        z1 = np.dot(noisy_image, weight_1) + bias_1
        a1 = np.where(z1 > 0, z1, z1 * 0.01)
        z2 = np.dot(a1, weight_2) + bias_2
        final_sigmoid = 1 / (1 + np.exp(-z2))

        reconstructed = final_sigmoid

        mse = np.mean((clean_image - reconstructed)**2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))

        zc = np.dot(a1, weight_class) + bias_class
        pred = np.argmax(zc, axis=1)
        accuracy = np.mean(pred == labels)
        print(f"Epoch {e} finished. Accuracy: {accuracy:.2f}. PSNR: {psnr:.2f}")





plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(clean_image[0].reshape(3,32,32).transpose(1,2,0))
plt.title("Original Picture")

plt.subplot(1,3,2)
plt.imshow(noisy_image[0].reshape(3,32,32).transpose(1,2,0))
plt.title("Noisy Picture")

plt.subplot(1,3,3)
output_image = np.clip(final_sigmoid[0], 0, 1).reshape(3, 32, 32).transpose(1, 2, 0)
plt.imshow(output_image)
plt.title("Denoised")
plt.show()
