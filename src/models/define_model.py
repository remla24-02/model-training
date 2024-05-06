"""
Define models for predictions.
"""

from keras.models import Sequential  # type: ignore # pylint: disable=import-error
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten  # type: ignore # pylint: disable=import-error
from joblib import dump, load


def _get_parameters():
    """
    Define parameters for models to train.
    """
    params = {'loss_function': 'binary_crossentropy',
              'optimizer': 'adam',
              'sequence_length': 200,
              'batch_train': 5000,
              'batch_test': 5000,
              'categories': ['phishing', 'legitimate'],
              'char_index': None,
              'epoch': 30,
              'embedding_dimension': 50,
              'dataset_dir': "../dataset/small_dataset/"}
    return params


def main():
    """
    Define the model and add the layers.
    """
    char_index = load('data/preprocessed/char_index.joblib')
    params = _get_parameters()

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

    dump(model, 'models/defined_model.joblib')


if __name__ == "__main__":
    main()
