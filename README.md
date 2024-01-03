# NumpyNet [WIP]
NumpyNet is a PyTorch-free framework for deep learning implemented with Numpy. 
![image](https://github.com/plarotta/NumpyNet/assets/20714356/3c41abed-eab8-4ece-8147-bc4688d6b894)

## Running
NumpyNet is lightweight by design. There are only a handful of libraries needed, and training should be relatively quick. You'll need Numpy, Pandas, matplotlib.pyplot, and tqdm, and you can install those onto your environment with the package manager of your choice (I am a big [Mamba](https://github.com/mamba-org/mamba) stan).

Train an example network using train_numpynet.py to ensure proper installation. From the root of this repo directory, run 

```python source/train_numpynet.py``` 

to train a small neural net on a predefined dataset. To use for your own projects you will need to refactor your dataset into the format expected by NumpyNet, but more documentation on this will come soon.
