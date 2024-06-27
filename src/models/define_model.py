"""
Define models for predictions.
"""
import random
import numpy as np
import tensorflow as tf
# type: ignore # pylint: disable=import-error
from keras.models import Sequential
# type: ignore # pylint: disable=import-error
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten
from joblib import dump, load


def _get_parameters(random_state=42, epoch=1, batch_size=5000):
    """
    Define parameters for models to train.
    """
    params = {'loss_function': 'binary_crossentropy',
              'optimizer': 'adam',
              'sequence_length': 200,
              'batch_train': batch_size,
              'batch_test': batch_size,
              'categories': ['phishing', 'legitimate'],
              'char_index': None,
              'epoch': epoch,
              'embedding_dimension': 50,
              'dataset_dir': "../dataset/small_dataset/",
              'random_state': random_state}
    return params


def main(params=None, model_name='defined_model', preprocessed_folder='data/preprocessed'):
    """
    Define the model and add the layers.
    """

    if params is None:
        params = _get_parameters()

    char_index = load(f'{preprocessed_folder}/char_index.joblib')

    np.random.seed(params['random_state'])
    tf.random.set_seed(params['random_state'])
    random.seed(params['random_state'])

    # Define models for training
    model = Sequential()
    voc_size = len(char_index.keys())
    print(f'voc_size: {voc_size}')

    model.add(Embedding(voc_size + 1, 50))

    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(len(params['categories']) - 1, activation='sigmoid'))

    dump(model, f'models/{model_name}.joblib')


if __name__ == "__main__":
    main()
