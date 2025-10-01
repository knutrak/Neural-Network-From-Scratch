import numpy as np

def sigmoid(z): #activation function
    return 1/(1+np.exp(-z))

def sigmoid_derivative(activation_l):
    return activation_l*(1-activation_l)

def softmax(z):
    z = z - np.max(z)           
    e = np.exp(z)
    return e / np.sum(e)


def binary_cross_entropy(pred, label):
    return -(label*np.log(pred) + (1-label)*np.log(1-pred))

def cross_entropy(p, y):
    
    return float(-np.sum(y*np.log(p)))

def standardize(X_train, X_test):
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def accuracy(pred, labels):

    comparison = (pred==labels)
    correct_pred = np.count_nonzero(comparison)
    return correct_pred/len(labels)


def train(model, X: np.array, y: np.array, loss_fn, lr):
    m = X.shape[0]
    predictions = []

    model.Delta = model.init_D()
    loss = 0
    for i in range(m):
        
        pred = model.forward_prop(X[i])
        d = model.calculate_small_d(y[i])  
        model.calculate_big_Delta(d)

        loss += loss_fn(pred, y[i])/m

        predictions.append(1 if pred>=0.5 else 0)


    grads = [Dl / m for Dl in model.Delta]
    model.update_weights(grads,lr=lr)

    acc = accuracy(np.array([predictions]).T, y)
    
        
    predictions = []

    return loss, acc


def test(NN, X_test, y_test):
    loss = 0
    predictions = []
    for i in range(len(X_test)):
        pred = NN.forward_prop(X_test[i])
        loss += 1/len(X_test) * pred

        predictions.append(1 if pred >=0.5 else 0)
    acc = accuracy(np.array([predictions]).T,y_test)

    return loss, acc


