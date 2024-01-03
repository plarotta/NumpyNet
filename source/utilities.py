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


def generate_dataset():
    data = pd.DataFrame(np.zeros((5000, 3)), columns=['x1', 'x2', 'y'])

    # Let's make up some noisy XOR data to use to build our binary classifier
    for i in range(len(data.index)):
        x1 = random.randint(0,1)
        x2 = random.randint(0,1)
        if x1 == 1 and x2 == 0:
            y = 0
        elif x1 == 0 and x2 == 1:
            y = 0
        elif x1 == 0 and x2 == 0:
            y = 1
        else:
            y = 2
        x1 = 1.0 * x1 + 0.20 * np.random.normal()
        x2 = 1.0 * x2 + 0.20 * np.random.normal()
        data.iloc[i,0] = x1
        data.iloc[i,1] = x2
        data.iloc[i,2] = y

    for i in range(int(0.25 *len(data.index))):
        k = np.random.randint(len(data.index)-1)
        data.iloc[k,0] = 1.5 + 0.20 * np.random.normal()
        data.iloc[k,1] = 1.5 + 0.20 * np.random.normal()
        data.iloc[k,2] = 1

    for i in range(int(0.25 *len(data.index))):
        k = np.random.randint(len(data.index)-1)
        data.iloc[k,0] = 0.5 + 0.20 * np.random.normal()
        data.iloc[k,1] = -0.75 + 0.20 * np.random.normal()
        data.iloc[k,2] = 2

    # Now let's normalize this data.
    data.iloc[:,0] = (data.iloc[:,0] - data['x1'].mean()) / data['x1'].std()
    data.iloc[:,1] = (data.iloc[:,1] - data['x2'].mean()) / data['x2'].std()

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    # The cost function is expecting numpy matrices so we need to convert X and y before we can use them.
    X = np.matrix(X.values)
    y = np.matrix(y.values)

    return(X,y)

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