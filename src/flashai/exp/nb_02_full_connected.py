
##########################################
###### This file was autogenerated. ######
######### DO NOT EDIT this file. #########
##########################################
### file to edit: dev_nb/imflash217__02_full_connected.ipynb ####

from exp.nb_01_matmul import *

def get_data(url=MNIST_URL):
    path = datasets.download_data(url, ext=".gz")
    with gzip.open(path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    return map(torch.tensor, (x_train, y_train, x_valid, y_valid))

def normalize(x, mean, std):
    return (x-mean)/std               ## using broadcasting to normalize

def test_near_zero(x, tol=1e-3):
    assert x.abs() < tol, f"Near zero: {x}"

from torch.nn import init

def mse(preds, target):
    """Mean Square Error loss"""
    return (preds.squeeze(-1)-target).pow(2).mean()