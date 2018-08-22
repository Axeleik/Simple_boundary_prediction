#import processing
#import neurofire.models as models
#from inferno.utils.io_utils import yaml2dict
from torch.nn.modules.module import _addindent
import torch
import numpy as np

def test_model_parameters():
    from train import load_Unet3D

    model = load_Unet3D({"train_config_folder": "train_config.yml"})

    size_of_model=0
    for param in model.parameters():
        print(param.dtype)
        size_of_model += np.prod(param.size())

    print("size of model (in parameters): ", size_of_model)

# Test

def extract_one_small_array():
    import os

    train_folder = os.path.join("../../to_copy/", "train/")
    print("loading raw_train...")
    raw_train = np.load(train_folder + "raw_train.npy")
    print("loading gt_train...")
    gt_train = np.load(train_folder + "gt_train.npy")
    print("Len(raw_train): ", len(raw_train))
    print("Len(gt_train): ", len(gt_train))

    np.save("../../to_copy/one_array_raw_gt.npy",(raw_train[0],gt_train[0]))

def load_sample_arrays(path = "../one_array_raw_gt.npy"):

    #has to be converted to float32 and divided by 255

    return np.load(path)



if __name__ == "__main__":
    raw, gt = load_sample_arrays()

    raw_array= [raw for i in range(5)]
    gt_array = [gt for i in range(5)]

    from blocks_dataset import blocksdataset
    from torch.utils.data import DataLoader

    data = blocksdataset(raw_array, gt_array)

    loader=DataLoader(data,batch_size=1,shuffle=True)
    test_model_parameters()


