import processing
import train

def main(config_dict):
    """
    main function (wrapper) for loading data (cropping, splitting) -> training
    :param config_dict: config
    """
    import os

    print("preparing data...")
    _, _, _, _, raw_test, gt_test = processing.load_crop_split_save_raw_gt(config_dict)

    print("preparing test loader")
    testloader = train.build_loader(raw_test, raw_test, batch_size=config_dict["batch_size_train"], shuffle=True)

    print("loading trained model...")
    import torch
    path_to_model = os.path.join(config_dict["project_folder"], "model/")
    best_checkpoint = torch.load(path_to_model + "best_checkpoint.pytorch")
    best_model = best_checkpoint.model

    print("preparing folder and files...")
    path_results = os.path.join(config_dict["project_folder"], "results/")
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    import h5py
    pred_results_file = h5py.File(path_results+"pred_results.h5", 'w')
    gt_file = h5py.File(path_results+"gt.h5", 'w')

    for i, data in enumerate(testloader, 0):

        print("prediction {} of {}".format(i, len(testloader)))

        raw, gt = data

        if torch.cuda.is_available():
            raw = raw.cuda()

        prediction = best_model(raw).squeeze(dim=0).cpu().detach().numpy()
        gt = gt.squeeze(dim=0).detach().numpy()

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
    print("Working with window_size {}, stride {}, ".format(config_dict["window_size"],
                                                    config_dict["stride"]))


    main(config_dict)
