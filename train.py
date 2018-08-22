import os
import processing



def main(config_dict):
    """
    main function (wrapper) for loading data (cropping, splitting) -> training
    :param config_dict: config
    """

    raw_train, gt_train, raw_val, gt_val, _, _ = processing.load_crop_split_save_raw_gt(config_dict)

    trainloader = build_loader(raw_train, gt_train, batch_size=config_dict["batch_size_train"], shuffle=True)
    valloader = build_loader(raw_val, gt_val, batch_size=config_dict["batch_size_val"], shuffle=False)

    U_net3D = load_Unet3D(config_dict)
    criterion, optimizer = get_criterion_and_optimizer(U_net3D, config_dict)

    train_net(config_dict, U_net3D, criterion, optimizer, trainloader, valloader)



def load_Unet3D(config_dict):
    """
    loads Unet3D with from neurofire
    :param config_dict: dictionary with all important paths
    :return: Unet3D model
    """

    import torch
    import neurofire.models as models
    from inferno.utils.io_utils import yaml2dict


    config = yaml2dict(config_dict["train_config_folder"])
    model_name = config.get('model_name')
    model = getattr(models, model_name)(**config.get('model_kwargs'))

    if torch.cuda.is_available():
        model.cuda()

    return model

def get_criterion_and_optimizer(net, config_dict):
    """
    Initializes criterion and optimizer for net
    :param net: NeuralNet
    :param config_dict: dictionary with all important paths
    :return: criterion and optimizer
    """

    import torch.nn as nn
    from inferno.utils.io_utils import yaml2dict
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()

    config = yaml2dict(config_dict["train_config_folder"])
    optimizer_kwargs = config.get('training_optimizer_kwargs')

    optimizer = optim.SGD(net.parameters(), lr=optimizer_kwargs.get('lr'), weight_decay=optimizer_kwargs.get('weight_decay'))

    return criterion, optimizer

def build_loader(raw, gt, batch_size=1, shuffle=True):
    """

    :param raw: list with raw_arrays
    :param gt: list with gt_arrays
    :param batch_size: size of batch
    :param shuffle: draw random each time
    :return: dataloader
    """

    from blocks_dataset import blocksdataset
    from torch.utils.data import DataLoader

    data = blocksdataset(raw, gt)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

def sorensen_dice_metric(prediction, target, eps=1e-6):
    """
    Computed the sorensen dice metric for validation
    :param prediction: predicted boundary image
    :param target: gt boundary image
    :param eps: min eps, so we do not divide by 0
    :return: metric score
    """

    assert prediction.size() == target.size()
    numerator = (prediction * target).sum()
    denominator = (prediction * prediction).sum() + (target * target).sum()

    return -2. * (numerator / denominator.clamp(min=eps))

def train_net(config_dict, net, criterion, optimizer, trainloader, valloader):
    """
    Trains the NeuralNet and saves the one with the best validation score
    :param config_dict: dict with configs
    :param net: NeuralNet
    :param criterion: criterion for NN
    :param optimizer: optimizer for NN
    :param trainloader: dataloader with traind ata
    :param valloader: dataloader with validation data
    """

    import torch

    model_folder = os.path.join(config_dict["project_folder"], "model/")
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    print("Start training!")

    best_val=0

    for epoch in range(config_dict["max_train_epochs"]):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            raw, gt = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(raw)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        #validation
        val_accumulated = 0.0

        for i, data in enumerate(valloader, 0):
            raw, gt = data
            outputs = net(raw)
            val_accumulated += sorensen_dice_metric(outputs, gt)

        print("Validation score after epoch {}: {}".format(epoch, val_accumulated))

        #save if better than best val score
        if val_accumulated>best_val:

            print("New best validation score: {}".format(val_accumulated))
            print("saving to ", model_folder + "best_model.torch")

            best_val=val_accumulated
            torch.save(net, model_folder + "best_model.torch")

    print('Finished Training')


if __name__ == "__main__":
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

    main(config_dict)
