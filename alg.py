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
lr = 0.5
labels = np.array(batch_1[b'labels'][:2000])
for e in range(200):
    
    layer_1=np.dot(noisy_image, weight_1) + bias_1
    relu = np.maximum(0, layer_1)
    layer_2=np.dot(relu, weight_2) + bias_2
    sigmoid= 1 / (1 + np.exp(-layer_2))

    class_val = np.dot(relu, weight_class) + bias_class
    expected_val = np.exp(class_val - np.max(class_val, axis=1, keepdims=True))
    probabilities = expected_val / np.sum(expected_val, axis=1, keepdims=True)

    predictions = np.argmax(probabilities, axis=1)
    accuracy = np.mean(predictions ==labels)

    correct_probabilities = -np.log(probabilities[range(2000), labels])
    class_loss = np.sum(correct_probabilities) / 2000

    grad_score = probabilities
    grad_score[range(2000), labels] -= 1
    grad_score /= 2000

    grad_weight_class = np.dot(relu.T, grad_score)
    grad_bias_class = np.sum(grad_score, axis=0, keepdims=True)

    relu = np.where(layer_1 > 0, layer_1, layer_1 * 0.01)
    grad_relu = np.where(layer_1 > 0, 1, 0.01)

    error_layer_2 = (sigmoid - clean_image) * (sigmoid * (1-sigmoid))
    error_layer_1 = (np.dot(error_layer_2, weight_2.T) + np.dot(grad_score, weight_class.T))* (layer_1 > 0)
    gradient_weight_2 = np.dot(relu.T, error_layer_2) / 2000
    gradient_bias_2 = np.sum(error_layer_2, axis=0, keepdims=True) / 2000
    gradient_weight_1 = np.dot(noisy_image.T, error_layer_1) / 2000
    gradient_bias_1 = np.sum(error_layer_1, axis=0, keepdims=True) / 2000

    weight_1 = weight_1 - (lr * gradient_weight_1)
    weight_2 = weight_2 - (lr * gradient_weight_2)
    bias_1 = bias_1 - (lr * gradient_bias_1)
    bias_2 = bias_2 - (lr * gradient_bias_2)

    

    weight_class -= lr * grad_weight_class
    bias_class -= lr * grad_bias_class

    
    mse = np.mean((clean_image - sigmoid)**2)
    psnr = 20 * np.log10(1.0/np.sqrt(mse))
    if e % 10 == 0:
        print(f"Epoch {e} finished. Accuracy: {accuracy:.2f}. PSNR: {psnr:.2f}")





plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(clean_image[0].reshape(3,32,32).transpose(1,2,0))
plt.title("Original Picture")

plt.subplot(1,3,2)
plt.imshow(noisy_image[0].reshape(3,32,32).transpose(1,2,0))
plt.title("Noisy Picture")

plt.subplot(1,3,3)
plt.imshow(sigmoid[0].reshape(3,32,32).transpose(1,2,0))
plt.title("Fixed Picture")

plt.show()