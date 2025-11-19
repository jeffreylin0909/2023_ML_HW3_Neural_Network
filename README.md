# 2023 ML HW3 Neural Network

This project is a practice of build Fully Connected Neural Network (FCNN) from scratch (only with numpy and math). 

The following Classes are implemented:
- Dense(self, n_x, n_y, seed=1): an n_x \times n_y Fully Connected Dense layer of FCNN.
- Activation(self, activation_function, loss_function, alpha=None, gamma=None): an activation layer of FCNN, if activation_function=="softmax", we treat it as output layer and return derivative of loss directly instead of activation, and if loss_function=="focal_loss", alpha and gamma is used.
- Model(self, units, activation_functions, loss_function, alpha=None, gamma=None): complete FCNN with len(units) layers, with first layer be input layer and last layer be output layer. The i-th layer has units\[i\] components, each with activation of activation_functions\[i-1\] (since input layer doesn't have activation).

Two dataset is tested:
- basic_data.npz: Predict patients’ health conditions, which get 90.00% accuracy on validation data.
- advanced_data.npz: MNIST handwritten digit dataset, which get 89.09% accuracy on validation data.

