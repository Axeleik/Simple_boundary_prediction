



def load_block(path):
    """
    This function loads a h5 (neuro) block
    :param path: path to block
    :return: block as np.array
    """

    from vigra.impex import readHDF5

    return readHDF5(path,"data")


def load_all_blocks(folder_path):
    """
    Returns array with all the blocks in one directory
    :param folder_path: path to folder with blocks
    :return: list with blocks
    """

    import os

    gt_files=os.listdir(folder_path)

    block_list=[]

    for file in gt_files:

        block_path=os.path.join(folder_path,file)

        block_list.append(load_block(block_path))

    return block_list


def extract_boundaries(img, connectivity=3):


    from skimage.segmentation import find_boundaries

    return find_boundaries(img, connectivity=connectivity).astype("int8")