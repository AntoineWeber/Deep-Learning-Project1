import numpy as np
import os
import errno

from six.moves import urllib

_default_root_url = 'https://documents.epfl.ch/users/f/fl/fleuret/www/data/bci'


def array_from_file(root, filename, base_url=_default_root_url) -> np.ndarray:
    """
    Returns a numpy array read from a File.

    :param str root: The path to the root data folder and location of the file.
    :param str filename: Complete name of the file to read.
    :param str base_url: Root URL where the files can be found.
    :return: An array loaded from file.
    :rtype: numpy.array
    """

    file_path = os.path.join(root, filename)

    if not os.path.exists(file_path):
        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        url = base_url + '/' + filename

        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        with open(file_path, 'wb') as f:
            f.write(data.read())

    return np.loadtxt(file_path)


def load(root, train=True, download=True, one_khz=False, nhwc=False, 
         normalize=False):
    """
    Args:

        root (string): Root directory of dataset.

        train (bool, optional): If True, creates dataset from training data.

        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

        one_khz (bool, optional): If True, creates dataset from the 1000Hz data instead
            of the default 100Hz.

        nhwc (bool, optional): If True, the layout of the returned tensor is
            nhwc (Batch, Height, Width, Chanel) which is the convention in
            Tensorflow, otherwise, BCHW is used, which is the convention in
            Pytorch. Default is False (Pytorch).
            
        normalize (bool, optional): If True, the input is normalize accross 
            each channel using the training data. 
    """

    nb_electrodes = 28

    if train or normalize:

        if one_khz:
            dataset = array_from_file(root, 'sp1s_aa_train_1000Hz.txt')
        else:
            dataset = array_from_file(root, 'sp1s_aa_train.txt')

        input = dataset[:, 1:].reshape(dataset.shape[0], nb_electrodes, -1)
        target = dataset[:, 0]
        
        # Training data mean and std on a per channel basis, used for 
        # normalization later
        mean = np.mean(input, axis=(0,2)).reshape(1, -1, 1);
        std = np.std(input, axis=(0,2)).reshape(1, -1, 1);

    else:

        if one_khz:
            input = array_from_file(root, 'sp1s_aa_test_1000Hz.txt')
        else:
            input = array_from_file(root, 'sp1s_aa_test.txt')
        target = array_from_file(root, 'labels_data_set_iv.txt')

        input = input.reshape(input.shape[0], nb_electrodes, -1)

    if normalize:
        input = input - mean;
        input = input/std;

    if nhwc:
        input = input.swapaxes(1, 2)

    return input, target


# Test code
if __name__ == "__main__":
    load("./data", train=True, one_khz=False, nhwc=True)
    load("./data", train=False, one_khz=False)
    load("./data", train=True, one_khz=True)
    load("./data", train=False, one_khz=True)
