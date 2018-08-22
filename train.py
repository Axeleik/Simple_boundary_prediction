import os
import processing


#6912 blocks
#weights=93662856
#weights.dtype=torch.float32

def main(paths_dict):

    import torch.optim as optim
    import torch.nn as nn
    raw_train, gt_train, raw_val, gt_val, _, _ = processing.load_crop_split_save_raw_gt(paths_dict)

    U_net3D = load_Unet3D(paths_dict)

    criterion, optimizer = get_criterion_and_optimizer(U_net3D, paths_dict)




    print("test")



def load_Unet3D(paths_dict):
    """
    loads Unet3D with from neurofire
    :param paths_dict: dictionary with all important paths
    :return: Unet3D model
    """
    import neurofire.models as models
    from inferno.utils.io_utils import yaml2dict


    config = yaml2dict(paths_dict["train_config_folder"])
    model_name = config.get('model_name')
    model = getattr(models, model_name)(**config.get('model_kwargs'))

    return model

def get_criterion_and_optimizer(net, paths_dict):
    """
    Initializes criterion and optimizer for net
    :param net: NeuralNet
    :param paths_dict: dictionary with all important paths
    :return: criterion and optimizer
    """

    import torch.nn as nn
    from inferno.utils.io_utils import yaml2dict
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()

    config = yaml2dict(paths_dict["train_config_folder"])
    optimizer_kwargs = config.get('training_optimizer_kwargs')

    optimizer = optim.SGD(net.parameters(), lr=optimizer_kwargs.get('lr'), weight_decay=optimizer_kwargs.get('weight_decay'))

    return criterion, optimizer

if __name__ == "__main__":

    paths_dict = {"blocks_folder_path": "../fib25_blocks",
    "raw_folder": "raw",
    "gt_folder": "gt",
    "project_folder": "../",
    "train_config_folder": "train_config.yml"}

    main(paths_dict)