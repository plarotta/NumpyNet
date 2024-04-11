import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from source.utilities import *
from source.models import MLP


def main():
    X = np.loadtxt("data/test_data_x.txt")
    y = np.loadtxt("data/test_data_y.txt")
    y = y.reshape((len(y),1))
    trainset, testset = test_train_split(X, y, 0.8)

    # create model architecture
    NN = MLP()
    NN.add_layer('Hidden', in_size=2, out_size=16)
    NN.add_layer('Hidden', in_size=16, out_size=16)
    NN.add_layer('Output', in_size=16, out_size=3)
    NN.add_layer('Loss', in_size=3, out_size=3)

    # train model
    loss = NN.train(trainset, epochs=40, lr=0.0005, batch_size=32, eval_data=testset, alpha=0.0)

    # plot training losses
    plt.plot(np.array(loss).flatten())
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.show()


if __name__ == '__main__':
    # main()
    NN = MLP()
    NN.add_layer('Hidden', in_size=5, out_size=64)
    NN.add_layer('Hidden', in_size=64, out_size=128)
    NN.add_layer('Output', in_size=128, out_size=2)
    NN.add_layer('Loss', in_size=2, out_size=2)
    for l in NN.layers:
        if l.layer_type != "L":
            print(l.weights.shape, l.biases.shape)


