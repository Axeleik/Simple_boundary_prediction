import os
import processing

def main(paths_dict):

    raw_train, gt_train, _, _ = processing.load_crop_split_save_raw_gt(paths_dict)

if __name__ == "__main__":

    paths_dict = {"blocks_folder_path": "../fib25_blocks",
    "raw_folder": "raw",
    "gt_folder": "gt",
    "project_folder": "../"}

    main(paths_dict)