from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    '''Base class for the layer subclasses.'''
    @abstractmethod
    def feedforward(self):
        pass
    def backprop(self):
        pass

class HiddenLayer(ABC):
    '''Class defining ff and backprop methods for hidden layers. 
    
    Attributes:
        weights (np.array)
        biases (np.array)
        in_dim (int): size of the input layer
        out_dim (int): size of the output layer (i.e. 2 for binary classification)
        layer_type (str): layer type identifier'''
    
    def __init__(self, in_size: int, out_size: int) -> None:
        self.weights = np.random.randn(in_size, out_size)
        self.biases = np.random.randn(out_size, 1)
        self.in_dim = in_size
        self.out_dim= out_size
        self.layer_type = "H"

    def relu(self, x: np.array) -> np.array:
        return np.maximum(0,x)

    def leaky_relu(self, x: np.array) -> np.array:
        alpha = 0.1
        return np.maximum(alpha*x, x)

    def leaky_relu_prime(self, x: np.array) -> np.array:
        '''derivative of leaky relu function'''
        x_copy = np.copy(x)
        x_copy[x_copy > 0] = 1.0
        x_copy[x_copy < 0] = 0.1
        return(x_copy)

    def relu_prime(self, x: np.array) -> np.array:
        '''derivative of relu function'''
        x_copy = np.copy(x)
        x_copy[x_copy > 0] = 1.0
        x_copy[x_copy < 0] = 0.0
        return(x_copy)

    def feedforward(self, x: np.array) -> np.array:
        Z = self.weights.T@x+self.biases
        activated_Z = self.relu(Z)
        return((Z, activated_Z))

    def backprop(self, 
                 layer_input: np.array, 
                 grad_h: np.array, 
                 h: np.array,
                 y: float) -> tuple[np.array, np.array, np.array]:
        f = self.relu_prime(h)
        grad_bias = np.multiply(f, grad_h)
        grad_weight = layer_input@(grad_bias.T)
        grad_h_to_pass =  self.weights@grad_h
        return((grad_weight, grad_bias, grad_h_to_pass))


class OutputLayer(ABC):
    '''Class defining ff and backprop methods for output layers. 
    
    Attributes:
        weights (np.array)
        biases (np.array)
        in_dim (int): size of the input layer
        out_dim (int): size of the output layer (i.e. 2 for binary classification)
        layer_type (str): layer type identifier'''
    
    def __init__(self, in_size: int, out_size: int) -> None:
        self.weights = np.random.randn(in_size, out_size)
        self.biases = np.random.randn(out_size, 1)
        self.in_dim = in_size
        self.out_dim = out_size
        self.layer_type = "O"

    def relu(self, x: np.array) -> np.array:
        return np.maximum(0,x)

    def leaky_relu(self, x: np.array) -> np.array:
        alpha = 0.1
        return np.maximum(alpha*x, x)

    def feedforward(self, x: np.array) -> tuple[np.array, np.array]:
        input_copy = np.copy(x)
        Z = self.weights.T@input_copy+self.biases
        activated_Z = np.exp(Z)/np.sum(np.exp(Z)+ 1e-11)
        assert Z.shape == activated_Z.shape, "output layer ff error"
        return((Z, activated_Z))

    def backprop(self, layer_input: np.array, grad_z: np.array, Z: np.array, 
                 y: float) -> tuple[np.array, np.array, np.array]:
        grad_weight = layer_input@(grad_z.T)
        grad_bias = np.copy(grad_z)

        grad_h = self.weights@grad_z

        return((grad_weight, grad_bias, grad_h))

class LossLayer(ABC):
    '''Class defining ff and backprop methods for loss layers. In this project 
    I embed the loss function into the network by making it a layer type of 
    its own.
    
    Attributes:
        in_dim (int): size of the input layer
        out_dim (int): size of the output layer (i.e. 2 for binary classification)
        layer_type (str): layer type identifier'''
    def __init__(self, in_size: int, out_size: int) -> None:
        self.in_dim = in_size
        self.out_dim= out_size
        self.layer_type = "L"

    def feedforward(self, y: float, z: np.array) -> float:
        '''cross-entropy loss'''
        loss = -z[int(y)] + np.log(np.sum(np.exp(z)) + 1e-11)
        return(loss)

    def backprop(self, Z: np.array, y: float) -> tuple[np.array]:
        y_hot = np.zeros((self.out_dim,1))
        y_hot[int(y)] = 1.0
        grad_Z = -1*y_hot + np.exp(Z)/np.sum(np.exp(Z) + 1e-11)
        return((grad_Z))