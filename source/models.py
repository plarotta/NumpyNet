from tqdm import tqdm
import numpy as np
from source.layers import *


class MLP():
    '''Multi-layer perceptron module. The model architecture is represented as the 
    list of layers self.layers, and the base operators for adding layers, inference,
    and backpropagation are defined below.
    
    Attributes:
        layers: A list containing the layers of the model. Each layer type has its own
        feedforward and backpropagation methods.
        trained: A boolean. 
    '''

    def __init__(self):
        self.layers = []
        self.trained = False

    def add_layer(self, 
                  layer_type: str, 
                  in_size: int, 
                  out_size: int) -> None:
      '''Method for adding layers to the model. Layers are defined as hidden, output,
      or loss, and the main reason for this is simply so we can treat the weights of 
      each layer appropriately. 
      
      Args:
          layer_type: A string (H, O, or L) denoting the layer type of the new layer.
          in_size: An integer defining the input size of the layer. Should be equal 
          to the out_size of the previous layer.
          out_size: An integer defining the output size of the layer. For hidden 
          layers, this out_size should equal the in_size of the next layer.
      '''

      # safety check to make sure a trained model is accidentally modified
      assert self.trained == False,\
          "Cannot modify the model architecture after it has already been trained"

      if layer_type in ("Hidden", "hidden", "H", "h"):
          layer = HiddenLayer(in_size, out_size)
      elif layer_type in ("Output", "output", "O", "o"):
          layer = OutputLayer(in_size, out_size)
      elif layer_type in ("Loss", "loss", "L", "l"):
          layer = LossLayer(in_size, out_size)
      else:
          raise ValueError('layer type must be one of \
                          ("Hidden","H","Output","O","Loss","L")')

      # update self.layers
      self.layers.append(layer)

    def net_feedforward(self, 
                        x: np.array) -> tuple[list,list]:
        '''Base feedforward method for an MLP network. The input is sent through
        each layer sequentially, and the activations & outputs at each are tracked
        and returned by this method at the end.
        
        Args:
            x: A np.array of the input to the network.
          
        Returns:
            A tuple containing the layer activations and the layer outputs generated
            from feedforward of the input.
        '''
        
        z_vals = [] # outputs of each layer are stored here
        activation_vals = [x] 
        a = np.copy(x)

        for layer in self.layers:
            if layer.layer_type != "L":
                z, a = layer.feedforward(a)
                z_vals.append(z)
                activation_vals.append(a)

        return((z_vals, activation_vals))

    def net_backprop(self, 
                     y: np.array, 
                     z_vals: list[np.array], 
                     activation_vals: list[np.array]) -> tuple[list,list]:
        '''Base backpropagation method for an MLP network. The loss is computed
        at the loss layer, and the gradients are propagated back through the layers
        all the way to the first layer. 
        
        Args:
            y: A np.array of the ground truth of the current input being passed 
            through the model.
            z_vals: A list containing all of the layer outputs from feedforward of 
            the current input.
            activation_vals: A list containing all of the layer activations from 
            feedforward of the current input.
          
        Returns:
            Tuple containing the gradients to the parameters of the network for 
            use in the update step. The first list in the output tuple contains 
            layer-wise gradients of the biases, and the second list contains the 
            layer-wise gradients of the weights.
        '''
      
        weight_deltas = []
        bias_deltas = []

        # loss layer
        grad_h = self.layers[-1].backprop(z_vals[-1], y) #grad_h is actually grad_z for loss layer

        # other layers
        for idx, layer in reversed(list(enumerate(self.layers[:-1]))):
            grad_weight, grad_bias, grad_h = layer.backprop(activation_vals[idx], 
                                                            grad_h, 
                                                            z_vals[idx], 
                                                            y)
            weight_deltas.append(grad_weight)
            bias_deltas.append(grad_bias)

        return((list(reversed(bias_deltas)), list(reversed(weight_deltas))))

    def process_mini_batch(self, 
                           mini_batch: list, 
                           lr: float, 
                           alpha: float) -> float:
        '''Method which runs the training routine on a single minibatch of data.
        Each sample is fed through the network, and then its gradients are 
        propagated back. The gradients are accumulated across the samples in the 
        minibatch, and the parameters are updated via a gradient step once the 
        feedforward/backprop routine has finished all of the samples. 
        
        Args:
            mini_batch: A minibatch of samples represented as a list of tuples 
            where each tuple (x,y) contains an observation x and its ground-truth
            label y.
            lr: Learning rate used in the gradient step.
            alpha: L2 regularization parameter for the weights.
          
        Returns:
            Average loss for the minibatch.
        '''
        
        first = True
        loss = 0.0
        for sample in mini_batch:
            x = sample[0].reshape(len(sample[0]),1)
            y = sample[1].reshape(len(sample[1]),1)

            if first:
                weights, activations = self.net_feedforward(x)
                bias_grad, weight_grad = self.net_backprop(y, weights, activations)

                loss = self.layers[-1].feedforward(y, weights[-1])
                first = False
            else:
                weights, activations = self.net_feedforward(x)
                t_bias_grad, t_weight_grad = self.net_backprop(y, weights, activations)
                loss += self.layers[-1].feedforward(y ,weights[-1])

                for i in range(len(bias_grad)):
                    bias_grad[i] = bias_grad[i] + t_bias_grad[i]

                for i in range(len(weight_grad)):
                    weight_grad[i] = weight_grad[i] + t_weight_grad[i]

        for i in range(len(weight_grad)):
            assert self.layers[i].weights.shape == weight_grad[i].shape, "W shape mismatch"
            self.layers[i].weights = self.layers[i].weights - lr/len(mini_batch)*(weight_grad[i]+alpha*np.abs(self.layers[i].weights))

        for i in range(len(bias_grad)):
            assert self.layers[i].biases.shape == bias_grad[i].shape, "B shape mismatch"

            self.layers[i].biases = self.layers[i].biases - lr/len(mini_batch)*bias_grad[i]
        return(loss/len(mini_batch))

    def train(self, 
              dataset: list, 
              epochs: int, 
              lr: float, 
              eval_data: list, 
              batch_size=32, 
              alpha=0.0) -> list:
        '''Method for training the network via minibatch gradient descent. 
        Progress bar reports training loss t_loss and validation accuracy v_acc.
        
        Args:
            dataset: A training dataset represented as a list of tuples (x,y)
            where x is the observation and y is the ground-truth label.
            epochs: An int defining how many times to pass through all of the 
            training data.
            lr: A float defining the learning rate.
            eval_data: A validation dataset represented as a list of tuples (x,y)
            where x is the observation and y is the ground-truth label.
            batch_size: Size of minibatches.
            alpha: L2 regularization parameter.

        Returns:
            Log of training losses.
        '''

        n = len(dataset)
        losses = []
        eval_acc = 0.0

        with tqdm(range(epochs)) as pbar:
          for j in pbar:
              np.random.shuffle(dataset)
              batch_loss = 0.0
              for k in range(n // batch_size):
                  mini_batch = dataset[k * batch_size : (k + 1) * batch_size]
                  loss = self.process_mini_batch(mini_batch, lr, alpha)
                  batch_loss += loss
              eval_acc = self.evaluate(eval_data)
              pbar.set_postfix({'train_loss':batch_loss[0]/batch_size, 'epoch':j+1, 'val_acc': eval_acc}, refresh=False)
              losses.append(batch_loss/batch_size)
        
        self.trained = True
        return losses

    def evaluate(self, 
                 val_data: np.array) -> float:
        test_results = [
            [np.argmax(self.net_feedforward(x.reshape(len(x),1))[-1][-1]), y[0]] for (x, y) in val_data
        ]
        res = np.array(test_results)
        return(np.sum(res[:,0] == res[:,1])/len(res))
