import os
import processing



def main(config_dict):
    """
    main function (wrapper) for loading data (cropping, splitting) -> training
    :param config_dict: config
    """

    print("preparing data...")
    raw_train, gt_train, raw_val, gt_val, _, _ = processing.load_crop_split_save_raw_gt(config_dict)

    print("preparing loaders...")
    trainloader = build_loader(raw_train, gt_train, batch_size=config_dict["batch_size_train"], shuffle=True)
    valloader = build_loader(raw_val, gt_val, batch_size=config_dict["batch_size_val"], val=True, shuffle=True)

    print("preparing Unet3D...")
    U_net3D = load_Unet3D(config_dict)
    criterion, optimizer = get_criterion_and_optimizer(U_net3D, config_dict)

    if not config_dict["process_only"]:

        if config_dict["inferno_train"]:
            train_net_with_inferno(config_dict, U_net3D, criterion, optimizer, trainloader, valloader)

        elif config_dict["debug"]:
            from test import do_one_loop
            do_one_loop(config_dict, U_net3D, criterion, optimizer, trainloader, valloader)

        else:
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

    from inferno.utils.io_utils import yaml2dict
    import torch.optim as optim
    from SorensenDiceLoss import SorensenDiceLoss

    criterion = SorensenDiceLoss()

    config = yaml2dict(config_dict["train_config_folder"])
    optimizer_kwargs = config.get('training_optimizer_kwargs')

    optimizer = optim.SGD(net.parameters(), lr=optimizer_kwargs.get('lr'),
                          weight_decay=optimizer_kwargs.get('weight_decay'))

    return criterion, optimizer

def build_loader(raw, gt, batch_size=1, shuffle=True, val=False):
    """

    :param raw: list with raw_arrays
    :param gt: list with gt_arrays
    :param batch_size: size of batch
    :param shuffle: draw random each time
    :return: dataloader
    """

    from blocks_dataset import blocksdataset
    from torch.utils.data import DataLoader

    data = blocksdataset(raw, gt, val=val)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=3)

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

    return - 2. * (numerator / denominator.clamp(min=eps))

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
    from time import time

    model_folder = os.path.join(config_dict["project_folder"], "model/")
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    print("Start training!")
    if config_dict["timestop"]:
        print("With timestop!")
    overall_time=time()

    best_val=0

    for epoch in range(config_dict["max_train_epochs"]):  # loop over the dataset multiple times

        running_loss = 0.0

        time_epoch=time()

        #so we record the dataloader time too
        if config_dict["timestop"]:
            time_iter = time()

        for i, data in enumerate(trainloader, 0):

            raw, gt = data

            if torch.cuda.is_available():
                raw = raw.cuda()
                gt = gt.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(raw).squeeze(dim=0)

            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()

            if config_dict["timestop"]:
                print("iteration {} took {} sec".format(i+1, time() - time_iter))
                time_iter = time()

            if config_dict["item"]:

                # print statistics
                running_loss += loss.item()

                if config_dict["timestop"]:
                    print("Loss: ", loss.item())

                if (i + 1) % 100 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

            if (i+1)%100==0:
                print("Finished iteration {} in {} min".format(i+1, (time() - time_epoch)/60 ))

        #validation
        val_accumulated = 0.0
        print("validating...")
        for j, data_val in enumerate(valloader, 0):
            raw, gt = data_val
            outputs = net(raw).squeeze(dim=0)
            outputs = outputs.detach()
            val_accumulated += sorensen_dice_metric(outputs, gt)
            if j==49:
                break

        print("")
        print("--------------------------------------------------------------------")
        print("Validation score after epoch {}: {}".format(epoch, val_accumulated))
        print("Best validation score: {}".format(best_val))

        #save if better than best val score
        if val_accumulated > best_val:

            print("saving to ", model_folder + "best_model.torch")

            best_val = val_accumulated
            torch.save(net, model_folder + "best_model.torch")

        print("{} hours passed".format((time() - overall_time) / 3600))

    print("saving last model...")
    torch.save(net, model_folder + "last_model.torch")
    print('Finished Training')





def train_net_with_inferno(config_dict, net, criterion, optimizer, trainloader, valloader):
    """
    Trains the NeuralNet with inferno
    :param config_dict: dict with configs
    :param net: NeuralNet
    :param criterion: criterion for NN
    :param optimizer: optimizer for NN
    :param trainloader: dataloader with traind ata
    :param valloader: dataloader with validation data
    """

    print("Start training with inferno!")

    from inferno.trainers.basic import Trainer
    from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore

    model_folder = os.path.join(config_dict["project_folder"], "model/")
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    trainer = Trainer(net) \
        .save_every((1, 'epochs'), to_directory=model_folder) \
        .build_criterion(criterion) \
        .build_optimizer(optimizer) \
        .build_metric(sorensen_dice_metric) \
        .evaluate_metric_every('never') \
        .validate_every((1, 'epochs'), for_num_iterations=50) \
        .register_callback(SaveAtBestValidationScore(smoothness=.5))

    trainer.set_max_num_epochs(config_dict['max_train_epochs'])
    trainer.bind_loader('train', trainloader).bind_loader('validate', valloader)
    trainer.cuda()
    trainer.fit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=int(160))
    parser.add_argument('--stride', type=int, default=int(90))
    parser.add_argument('--clear', type=bool, default=False)
    parser.add_argument('--max_train_epochs', type=int, default=int(200))
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--process_only', type=bool, default=False)
    parser.add_argument('--timestop', type=bool, default=False)
    parser.add_argument('--inferno_train', type=bool, default=True)

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
        "max_train_epochs": args.max_train_epochs,
        "debug": args.debug,
        "process_only": args.process_only,
        "item": False,
        "timestop": args.timestop,
        "inferno_train": args.inferno_train}

    print("Starting...")
    print("Working with window_size {}, stride {}, "
          "and a maximum train epochs of {}".format(config_dict["window_size"],
                                                    config_dict["stride"],
                                                    config_dict["max_train_epochs"]))
    if config_dict["process_only"]:
        print("Only data processing")

    main(config_dict)
