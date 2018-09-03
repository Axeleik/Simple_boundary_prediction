#import processing
#import neurofire.models as models
#from inferno.utils.io_utils import yaml2dict
from torch.nn.modules.module import _addindent
import torch
import numpy as np
from processing import extract_boundaries
from train import load_Unet3D, get_criterion_and_optimizer

def test_model_parameters():
    from train import load_Unet3D

    model = load_Unet3D({"train_config_folder": "train_config.yml"})

    size_of_model=0
    for param in model.parameters():
        print(param.dtype)
        size_of_model += np.prod(param.size())

    print("size of model (in parameters): ", size_of_model)

# Test

def do_one_loop(config_dict, net, criterion, optimizer, trainloader, valloader):

    import torch
    import gc
    from time import time
    import os
    from train import sorensen_dice_metric

    model_folder = os.path.join(config_dict["project_folder"], "model/")
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    print("Start training with TEST!")
    overall_time=time()

    best_val=0

    for epoch in range(config_dict["max_train_epochs"]):  # loop over the dataset multiple times

        running_loss = 0.0
        time_start = time()

        for i, data in enumerate(trainloader, 0):

            raw, gt = data
            print ("raw.size: {}, raw.dtype: {}".format(raw.size(), raw.dtype))
            print ("gt.size: {}, gt.dtype: {}".format(gt.size(), gt.dtype))


            time_load_data = time()

            print("Time to load data: ", time_load_data - time_start, " secs")

            if torch.cuda.is_available():
                gt = gt.cuda()
                time_gt = time()

                raw = raw.cuda()

            time_cuda=time()
            print("Time to cuda gt: ", time_gt - time_load_data, " secs")
            print("Time to cuda raw: ", time_cuda - time_gt , " secs")

            # zero the parameter gradients
            optimizer.zero_grad()
            time_zero_grad=time()
            print("Time zero_grad: ",time_zero_grad-time_cuda, " secs")

            # forward + backward + optimize
            outputs = net(raw).squeeze(dim=0)
            time_outputs=time()
            print("Time to output: ",time_outputs-time_zero_grad, " secs")

            loss = criterion(outputs, gt)
            time_loss=time()
            print("Time for loss: ",time_loss-time_outputs, " secs")

            loss.backward()
            time_backward = time()
            print("Time for backward: ",time_backward-time_loss, " secs")

            optimizer.step()
            time_optimizer_step=time()
            print("Time for optimizer: ",time_optimizer_step-time_backward, " secs")

            ## print statistics
            #running_loss += loss.item()

            time_delete=time()
            print("time for loss: ",time()-time_optimizer_step," secs")

            if torch.cuda.is_available():
                #gt = gt.cpu()
                time_gt = time()

                #raw = raw.cpu()
                del raw
                del gt
                torch.cuda.empty_cache()
                gc.collect()

            print("time for cpu gt: ", time_gt-time_delete)
            print("time for cpu raw: ", time()-time_gt)


            print("time for iteration: ", time()-time_start," secs")
            time_start=time()
            print("-------------------------------------------")


            if i==10:
                break
        break
        val_accumulated=0.0
        print("now starting val...")
        for j, data_val in enumerate(valloader, 0):
            raw, gt = data_val
            outputs = net(raw).squeeze(dim=0)
            outputs = outputs.detach()
            val_accumulated += sorensen_dice_metric(outputs, gt)
            print("val_acc:  {}".format(val_accumulated))

            if j==3:
                assert (1==2),"stop!"


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

def load_sample_arrays(path = "/HDD/embl/fib25_blocks/one_array_raw_gt.npy"):

    #has to be converted to float32 and divided by 255

    return np.load(path)

def save_to_h5(path_in,folder_out):
    import numpy as np
    import h5py
    gt="/net/hci-storage02/userfolders/amatskev/simple_boundary_prediction_project_folder/train/gt_train_w160_s90.npy"
    raw="/net/hci-storage02/userfolders/amatskev/simple_boundary_prediction_project_folder/train/raw_train_w160_s90.npy"

    a=np.load(path_in)

    for idx,i in enumerate(a):
        f=h5py.File(folder_out+"{}.h5".format(idx), 'w')
        f.create_dataset('data', data=i)
        f.close()


if __name__ == "__main__":


    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=int(80))
    parser.add_argument('--stride', type=int, default=int(40))
    parser.add_argument('--clear', type=bool, default=False)
    parser.add_argument('--max_train_epochs', type=int, default=int(15))

    args = parser.parse_args()


    config_dict = {
        "project_folder": "/net/hci-storage02/userfolders/amatskev/simple_boundary_prediction_project_folder/",
        "clear": args.clear,
        "raw_folder": "fib25_blocks/raw/",
        "gt_folder": "fib25_blocks/gt/",
        "train_config_folder": "train_config.yml",
        "window_size": args.window_size,
        "stride": args.stride,
        "batch_size_train": 1,
        "batch_size_val": 1,
        "max_train_epochs": args.max_train_epochs}


    raw, gt = load_sample_arrays()

    raw_array= np.array([raw for i in range(5)])
    gt_array = np.array([gt for i in range(5)])
    gt_blocks_all = list(map(lambda x: extract_boundaries(x), gt_array))

    from blocks_dataset import blocksdataset
    from torch.utils.data import DataLoader

    data = blocksdataset(raw_array, gt_blocks_all)
    model = load_Unet3D(config_dict)
    loader = DataLoader(data,batch_size=1,shuffle=True)
    criterion, optimizer = get_criterion_and_optimizer(model, config_dict)

    for i, data in enumerate(loader, 0):
        raw, gt = data
        outputs = model(raw)
        loss = criterion(outputs, gt)
        loss.backward()
        optimizer.step()
    test_model_parameters()
    """

