import processing
from train import sorensen_dice_metric
import train


def main(config_dict):
    """
    main function (wrapper) for loading data (cropping, splitting) -> training
    :param config_dict: config
    """
    import os

    print("preparing data...")
    raw_test, gt_test = processing.load_crop_split_save_raw_gt(config_dict, False)

    print("preparing test loader")
    testloader = train.build_loader(raw_test, gt_test, batch_size=config_dict["batch_size_train"], shuffle=True)

    print("loading trained model...")
    import torch
    path_to_model = os.path.join(config_dict["project_folder"], "model/")
    best_checkpoint = torch.load(path_to_model + "best_checkpoint.pytorch")
    print("Keys in checkpoint: ",best_checkpoint.keys())
    best_model = best_checkpoint["_model"]

    print("preparing folder and files...")
    path_results = os.path.join(config_dict["project_folder"], "results/")
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    import h5py

    threshold = config_dict["threshold"]
    for i, data in enumerate(testloader, 0):

        print("prediction {} of {}".format(i, len(testloader)))

        raw, gt = data

        if torch.cuda.is_available():
            raw = raw.cuda()

        prediction = best_model(raw).squeeze().cpu().detach().numpy()
        prediction[prediction>=threshold]=1
        prediction[prediction!=1] = 0
        gt = gt.squeeze().detach().numpy()
        #print("prediction.shape: ",prediction.shape)
        #print("gt.shape: ", gt.shape)

        pred_results_file = h5py.File(path_results + "pred_results_{}.h5".format(i), 'w')
        gt_file = h5py.File(path_results + "gt_{}.h5".format(i), 'w')
        pred_results_file.create_dataset('pred_{}'.format(i), data=prediction, compression="gzip", compression_opts=9)
        gt_file.create_dataset('gt_{}'.format(i), data=gt, compression="gzip", compression_opts=9)
        pred_results_file.close()
        gt_file.close()

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
    parser.add_argument('--threshold', type=float, default=0.6)

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
        "inferno_train": args.inferno_train,
        "threshold": args.threshold}

    print("Starting...")
    print("Working with window_size {}, stride {}, ".format(config_dict["window_size"],
                                                    config_dict["stride"]))


    main(config_dict)
