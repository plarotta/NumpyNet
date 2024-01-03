import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import *
from models import MLP



if __name__ == '__main__':
    # dataset generation
    X,y = generate_dataset()
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
