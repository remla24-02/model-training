"""
Preprocess the data to be trained by the learning algorithm.
"""

import os

from lib_ml_remla24_team02 import data_preprocessing


def main():
    """
    Preprocess data by using library lib-ml.
    """

    data_preprocessing.preprocess(os.path.join(
        'data', 'raw'), os.path.join('data', 'preprocessed'))


if __name__ == '__main__':
    main()
