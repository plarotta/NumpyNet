from source.models import *

def test_model_generation():
    '''simple test to check that adding layers to an MLP model object works'''

    NN = MLP()
    NN.add_layer('Hidden', in_size=2, out_size=16)
    NN.add_layer('Output', in_size=16, out_size=3)
    NN.add_layer('Loss', in_size=3, out_size=3)

    assert True == True, 'dummy message'

def test_layer_sizes():
    '''simple test to check the size of the weights and biases at each layer'''

    NN = MLP()
    NN.add_layer('Hidden', in_size=5, out_size=64)
    NN.add_layer('Hidden', in_size=64, out_size=128)
    NN.add_layer('Output', in_size=128, out_size=2)
    NN.add_layer('Loss', in_size=2, out_size=2)
    sizes = []
    for l in NN.layers:
        if l.layer_type != "L":
            sizes.append((l.weights.shape, l.biases.shape))
    expected_sizes = [((5, 64),(64, 1)),((64, 128),(128, 1)),((128, 2),(2, 1))]
    
    assert sizes == expected_sizes, 'model layer weights an biases are wrong shape'
