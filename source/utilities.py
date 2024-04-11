import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


def test_train_split(x_data, y_data, train_fraction = 0.8):
    dataset = [(x,y) for x,y in zip(np.array(x_data), np.array(y_data))]
    np.random.shuffle(dataset)
    split_idx = int(len(dataset)*train_fraction)
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]
    assert len(train_set) + len(test_set) == len(dataset), "lengths of data splits are inconsistent"
    return(train_set,test_set)


def check_decision_boundary(trained_net):
    points = np.random.uniform(-2,2,(75000,2))

    d = np.argmax(np.array([trained_net.net_feedforward(p.reshape(2,1))[-1][-1] for p in points]),axis=1).flatten()
    scatter = plt.scatter(points[:,0], points[:,1], c=d, s=4)
    plt.legend(handles=scatter.legend_elements()[0], labels=["0","1","2"])

    plt.title(" MLP Decision Boundary")
    return(d)


def plot_data(X, y_predict):

    fig, ax = plt.subplots(figsize=(12,8))
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    indices_0 = [k for k in range(0, X.shape[0]) if y_predict[k] == 0]
    indices_1 = [k for k in range(0, X.shape[0]) if y_predict[k] == 1]
    indices_2 = [k for k in range(0, X.shape[0]) if y_predict[k] == 2]

    ax.plot(X[indices_0, 0], X[indices_0,1], marker='o', linestyle='', ms=5, label='0')
    ax.plot(X[indices_1, 0], X[indices_1,1], marker='o', linestyle='', ms=5, label='1')
    ax.plot(X[indices_2, 0], X[indices_2,1], marker='o', linestyle='', ms=5, label='2')

    ax.legend()
    ax.legend(loc=2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Tricky 3 Class Classification')
    plt.show()