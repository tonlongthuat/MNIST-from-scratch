import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from neural_network import NN
import gradio as gr


data_train=pd.read_csv('train.csv')
# print(data_train.head(5))
y_train = np.array(data_train.iloc[:, 0]) #take the first column as the label
X_train = np.array(data_train.iloc[:, 1:])
# print(X_train)

# Splitting the data into train and test sets
num_samples = X_train.shape[0]
num_test_samples = int(num_samples * 0.2) # 20% of the data for testing
# Randomly select indices for test set
test_indices = np.random.choice(num_samples, num_test_samples, replace=False)
# Create train and test sets
X_test = X_train[test_indices]
y_test = y_train[test_indices]
#delete test data from train data by row
X_train = np.delete(X_train, test_indices, axis=0) 
y_train = np.delete(y_train, test_indices, axis=0)

# normalize
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.T # (784, 33600)
X_test = X_test.T # (784, 8400)

# def one_hot(y):
#     n_classes = len(np.unique(y_train))
#     y_oh = np.eye(n_classes)
#     return y_oh.T
# a=one_hot(y_train)[[5]]
# print(a)

#test mini batch
# m=X_train.shape[1]
# permutation = list(np.random.permutation(m)) 
# X_shuffled = X_train[:, permutation]
# y_shuffled = y_train[permutation]
# mini_batch_size=3
# k=1
# mini_batch=[]
# mini_batch_X = X_shuffled[:, k*mini_batch_size:(k+1)*mini_batch_size]
# mini_batch_Y = y_shuffled[k*mini_batch_size:(k+1)*mini_batch_size]
# mini_batch.append((mini_batch_X, mini_batch_Y))

# print img
# index=20
# img=X_train[:,index].reshape(28,28)*255
# plt.gray()
# plt.imshow(img, interpolation='nearest')
# print(y_train[index])
# plt.show()

nn = NN(X_train, y_train)
nn.fit(X_train,y_train,512, 0.01, 100,'save_model/my_model')
nn.load_model('./save_model/my_model.npz')

# Make a prediction with the first image
# prediction = nn.predict(X_test_first_image)
# print(prediction)
# prediction=nn.predict(X_train)
# acc=nn.accuracy(prediction,y_train)
# print(acc)

# def digit_recognizer(image):
#     # Reshape the image from (28, 28) to (784,)
#     image = image.reshape(-1,1)
#     image = image.astype('float32') / 255.
#     # Make a prediction with the neural network
#     prediction = nn.predict(image)
#     return  prediction
    

# iface = gr.Interface(
#     fn=digit_recognizer,
#     inputs=gr.Image(shape=(28,28), image_mode='L', invert_colors=True, source="canvas"),
#     # outputs=[gr.Image(shape=(28,28)), gr.outputs.Textbox()]
#     outputs=gr.outputs.Textbox(),
# )
# iface.launch()


