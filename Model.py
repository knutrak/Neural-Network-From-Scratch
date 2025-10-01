import numpy as np
from utils import sigmoid, sigmoid_derivative, softmax


class NeuralNet:
    

    def __init__(self, n_inputs, hidden_layers, n_output, output = "sigmoid"):
        self.layer_sizes = [n_inputs] + hidden_layers + [n_output]
        self.output = output
        
        self.W = [] # Weights: each layer has a bias column (+1)
        for l in range(len(self.layer_sizes)-1):
            fan_in  = self.layer_sizes[l]
            fan_out = self.layer_sizes[l+1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            Wl = np.random.uniform(-limit, limit, size=(fan_out, fan_in + 1))  # +1 for bias col
            self.W.append(Wl)
        
    
        self.activations = [np.zeros((n,1)) for n in self.layer_sizes]

        self.Delta = self.init_D()


    def forward_prop(self, x: np.array):
        a = x.reshape(-1,1)
        self.activations[0] = a
        L = len(self.W)

        for l in range(L-1): #Sigmoid activation for hidden layers
            a_tilde = np.vstack([np.ones((1,1)), a])
            z = self.W[l] @ a_tilde
            a = sigmoid(z)
            self.activations[l+1] = a

        
        a_tilde = np.vstack([np.ones((1,1)), a])
        zL = self.W[-1] @ a_tilde
        if self.output == "softmax":
            aL = softmax(zL)      # probabilities that sum to 1
        else:
            aL = sigmoid(zL)
        self.activations[-1] = aL
        return self.activations[-1]

    def update_weights(self, grads, lr):
        for l in range(len(self.W)):
            self.W[l] = self.W[l] - lr*grads[l]

    def calculate_small_delta(self, labels: np.array):
        y = labels.reshape(-1,1)
        L = len(self.layer_sizes) - 1  # last layer index
        d = [np.zeros((n,1)) for n in self.layer_sizes]

        # output layer
        d[L] = self.activations[L] - y                  # (n_L,1)

        for l in range(L-1, 0, -1):#Iterating backwards
            W_no_bias = self.W[l][:, 1:]           # (n_l, n_{l+1})
            a_l = self.activations[l]                  # (n_l,1)
            d[l] = (W_no_bias.T @ d[l+1]) * sigmoid_derivative(a_l) 
        return d
    
    def calculate_big_Delta(self, d):
        for l in range(len(self.W)):
            a_l = self.activations[l] #(n_l,1)
            a_tilde = np.vstack([np.ones((1,1)), a_l])  #(n_l+1,1)
            self.Delta[l] += d[l+1] @ a_tilde.T



    def init_D(self):
        return [np.zeros_like(Wl) for Wl in self.W] # Same shapes as W (including bias column)



    def __str__(self):
        string = f"My Neural Network\n-------------------------\nInput size: {self.layer_sizes[0]}\n"
        for i in range(1,len(self.layer_sizes)-1):
            string += f"Nodes layer {i+1}: {self.layer_sizes[i]}\n"
        string += f"Nodes output:  {self.layer_sizes[-1]}\n\n"

        string += "-------------------------\n"

        for layer_idx, W_l in enumerate(self.W):
            string += f"Shape W_{layer_idx} = {self.W[layer_idx].shape} | Shape a_{layer_idx} = {self.activations[layer_idx].shape}\n"
            string += f"Adding bias: a_{layer_idx} = [1, a_{layer_idx}].T\n"
            z = self.W[layer_idx]@np.vstack([np.ones((1,1)), self.activations[layer_idx]])
            string+= f"z_{layer_idx+1} = W_{layer_idx} @ a_{layer_idx} | Shape = {z.shape}\n"
            string += f"a_{layer_idx+1} = [1, g([z_{layer_idx+1}])] | Shape = {self.activations[layer_idx+1].shape}\n\n"
        return string
