# NumpyNet [WIP]
NumpyNet is a Numpy-only multi-layer perceptron (MLP) implementation in Python. _It is not intended to compete with PyTorch/Keras/TensorFlow. It is just for educational purposes._
![image](https://github.com/plarotta/NumpyNet/assets/20714356/3c41abed-eab8-4ece-8147-bc4688d6b894)

## Install 
After cloning the repo, create a conda (or mamba) environment with Python 3.11:
```conda create -n numpynet-env python=3.11```

Activate environment:
```conda activate numpynet-env```

Install dependencies:
```conda install numpy matplotlib pytest tqdm```

## Running
From the root of the repo, run:
```python train.py```

## Testing
From the root of the repo, run:
```pytest```