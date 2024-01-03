from tqdm import tqdm
import numpy as np


class MLP():
  def __init__(self, test_data):
    self.layers = []
    self.test_data = test_data

  def add_layer(self, layer_type: str, in_size: int, out_size: int):
    if layer_type in ("Hidden", "hidden", "H", "h"):
      layer = HiddenLayer(in_size, out_size)
    elif layer_type in ("Output", "output", "O", "o"):
      layer = OutputLayer(in_size, out_size)
    elif layer_type in ("Loss", "loss", "L", "l"):
      layer = LossLayer(in_size, out_size)

    self.layers.append(layer)

  def net_feedforward(self, x):
    z_vals = []
    activation_vals = [x]
    a = np.copy(x)
    for layer in self.layers:
      if layer.layer_type != "L":
        z, a = layer.feedforward(a)
        z_vals.append(np.copy(z))
        activation_vals.append(np.copy(a))
    return((z_vals, activation_vals))

  def net_backprop(self, x, y, z_vals, activation_vals):
    weight_deltas = []
    bias_deltas = []

    layer_input = np.copy(x.reshape(len(x),1))

    # loss layer
    grad_h = self.layers[-1].backprop(z_vals[-1], y) #grad_h is actually grad_z for loss layer

    # other layers
    for idx, layer in reversed(list(enumerate(self.layers[:-1]))):
      grad_weight, grad_bias, grad_h = layer.backprop(activation_vals[idx], np.copy(grad_h), z_vals[idx], y)
      weight_deltas.append(np.copy(grad_weight))
      bias_deltas.append(np.copy(grad_bias))

    return((list(reversed(bias_deltas)), list(reversed(weight_deltas))))

  def process_mini_batch(self, mini_batch, lr, alpha):
    first = True
    loss = 0.0
    for sample in mini_batch:
      x = sample[0].reshape(len(sample[0]),1)
      y = sample[1].reshape(len(sample[1]),1)

      if first:
        weights, activations = self.net_feedforward(x)
        bias_grad, weight_grad = self.net_backprop(x, y, weights, activations)

        loss = self.layers[-1].feedforward(y, weights[-1])
        first = False
      else:
        weights, activations = self.net_feedforward(x)
        t_bias_grad, t_weight_grad = self.net_backprop(x, y, weights, activations)
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

  def train(self, dataset, epochs, lr, batch_size=32, test=False, alpha=0.0):
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
          eval_acc = self.evaluate(self.test_data)
          pbar.set_postfix({'t_loss':batch_loss[0]/batch_size, 'epoch':j+1, 'v_acc': eval_acc}, refresh=False)
          losses.append(batch_loss/batch_size)

    return losses

  def evaluate(self, val_data):

    test_results = [
        [np.argmax(self.net_feedforward(x.reshape(len(x),1))[-1][-1]), y[0]] for (x, y) in val_data
    ]
    res = np.array(test_results)
    return(np.sum(res[:,0] == res[:,1])/len(res))

class HiddenLayer():
  def __init__(self, in_size, out_size):
    self.weights = np.random.randn(in_size, out_size)
    self.biases = np.random.randn(out_size, 1)
    self.in_dim = in_size
    self.out_dim= out_size
    self.layer_type = "H"

  def relu(self, x):
    return np.maximum(0,x)

  def leaky_relu(self, x):
    alpha = 0.1
    return np.maximum(alpha*x, x)

  def leaky_relu_prime(self, x):
    x_copy = np.copy(x)
    x_copy[x_copy > 0] = 1.0
    x_copy[x_copy < 0] = 0.1
    return(x_copy)

  def relu_prime(self,x):
    x_copy = np.copy(x)
    x_copy[x_copy > 0] = 1.0
    x_copy[x_copy < 0] = 0.0
    return(x_copy)

  def feedforward(self, x):

    Z = self.weights.T@x+self.biases
    activated_Z = self.relu(Z)

    return((Z, activated_Z))

  def backprop(self, layer_input, grad_h, h, y):
    f = self.relu_prime(h)

    grad_bias = np.multiply(f, grad_h)

    grad_weight = layer_input@(grad_bias.T)

    grad_h_to_pass =  self.weights@grad_h

    return((grad_weight, grad_bias, grad_h_to_pass))


class OutputLayer():
  def __init__(self, in_size, out_size):
    self.weights = np.random.randn(in_size, out_size)
    self.biases = np.random.randn(out_size, 1)
    self.in_dim = in_size
    self.out_dim = out_size
    self.layer_type = "O"

  def relu(self, x):
    return np.maximum(0,x)

  def leaky_relu(self, x):
    alpha = 0.1

    return np.maximum(alpha*x, x)

  def feedforward(self, x: np.array):
    input_copy = np.copy(x)

    Z = self.weights.T@input_copy+self.biases
    activated_Z = np.exp(Z)/np.sum(np.exp(Z)+ 1e-11)
    assert Z.shape == activated_Z.shape, "output layer ff error"

    return((Z, activated_Z))

  def backprop(self, layer_input, grad_z, Z, y):

    grad_weight = layer_input@(grad_z.T)
    grad_bias = np.copy(grad_z)

    grad_h = self.weights@grad_z

    return((grad_weight, grad_bias, grad_h))

class LossLayer():
  def __init__(self, in_size, out_size):
    self.in_dim = in_size
    self.out_dim= out_size
    self.layer_type = "L"

  def feedforward(self, y, z):
    loss = -z[int(y)] + np.log(np.sum(np.exp(z)) + 1e-11)
    return(loss)

  def backprop(self, Z, y):
    y_hot = np.zeros((self.out_dim,1))
    y_hot[int(y)] = 1.0

    grad_Z = -1*y_hot + np.exp(Z)/np.sum(np.exp(Z) + 1e-11)

    return((grad_Z))