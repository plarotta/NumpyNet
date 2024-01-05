# NumpyNet [WIP]
NumpyNet is a Numpy-only deep learning framework for Python. _It is not intended to compete with PyTorch/Keras/TensorFlow. It is mostly for educational purposes._
![image](https://github.com/plarotta/NumpyNet/assets/20714356/3c41abed-eab8-4ece-8147-bc4688d6b894)

## Running
NumpyNet is lightweight. There are only a handful of libraries needed, and training should be relatively quick. You'll need Numpy, Pandas, matplotlib.pyplot, and tqdm, and you can install those onto your environment with the package manager of your choice (I am a big [Mamba](https://github.com/mamba-org/mamba) stan).

Train an example network using train_numpynet.py to ensure proper installation. From the root of this repo directory, run 

```python source/train_numpynet.py``` 

to train a small neural net on a predefined dataset. To use for your own projects you will need to refactor your dataset into the format expected by NumpyNet, but more documentation on this will come soon.

## Example
To show NumpyNet in action, I created a relatively difficult dataset to fit to. The dataset consists of 5000 samples described by 2 features and belonging to 3 different classes. As you can see below, the decision boundaries should be non-linear.
![image](https://github.com/plarotta/NumpyNet/assets/20714356/5222eb16-bb23-43e0-9ff4-65134372aee4)

With the observations _X_ and the labels _y_, we can now do a train-test split so we can move forward with model training:

```trainset, testset = test_train_split(X,y, 0.8)```

Next step is to initialize our model. We do this by creating an MPL() instance and then adding layers to it:

```NN = MLP(testset)
NN.add_layer('Hidden', in_size=2, out_size=16)
NN.add_layer('Hidden', in_size=16, out_size=16)
NN.add_layer('Output', in_size=16, out_size=3)
NN.add_layer('Loss', in_size=3, out_size=3)
```

Finally we can move ahead to training:

```loss = NN.train(trainset, epochs=40, lr=0.0005, batch_size=32, test=True,alpha=0.0)```

Our model reached a 93% accuracy on the test dataset, which is not bad! With some hyperparameter tuning and some regularization we can reach 98%+. Lastly, we can visualize our training losses to get a general sense of model fit, and then plot points to visualize our model's decision boundary:

```plt.plot(np.array(loss).flatten())
plt.xlabel("batch")
plt.ylabel("loss")
plt.title("Training Loss")

def check_decision_boundary(trained_net):
  points = np.random.uniform(-2,2,(75000,2))

  d = np.argmax(np.array([trained_net.net_feedforward(p.reshape(2,1))[-1][-1] for p in points]),axis=1).flatten()
  scatter = plt.scatter(points[:,0], points[:,1], c=d, s=4)
  plt.legend(handles=scatter.legend_elements()[0], labels=["0","1","2"])

  plt.title(" MLP Decision Boundary")
  return(d)

check_decision_boundary(NN)
```

<img width="595" alt="image" src="https://github.com/plarotta/NumpyNet/assets/20714356/a1930497-4ea8-4c81-b63c-1c144f9032a8">

![image](https://github.com/plarotta/NumpyNet/assets/20714356/2a602e7b-7e12-4c5c-9c02-ef97196c85c3)

