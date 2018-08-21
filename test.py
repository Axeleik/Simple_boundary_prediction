import processing

if __name__ == "__main__":

    test_block_path = "/HDD/embl/fib25_blocks/raw/raw_block1.h5"
    test_blocks_folder = "/HDD/embl/fib25_blocks/raw"

    test_block = processing.load_block(test_block_path)

    print("Test block type: ", type(test_block))
    print("Test block shape: ", test_block.shape)
    print("Test block dtype: ", test_block.dtype)

    print("_")
