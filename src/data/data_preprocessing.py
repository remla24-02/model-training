"""
Preprocess the data to be trained by the learning algorithm.
"""

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore # pylint: disable=import-error
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore # pylint: disable=import-error
from joblib import dump


def _load_data():
    """
    Loading the dataset.
    """
    with open('./data/raw/train.txt', 'r', encoding='utf-8') as train_file:
        train_lines = train_file.readlines()[1:]
        raw_x_train = [line.split("\t")[1].strip() for line in train_lines]
        raw_y_train = [line.split("\t")[0].strip() for line in train_lines]

    with open('./data/raw/test.txt', 'r', encoding='utf-8') as test_file:
        test_lines = test_file.readlines()
        raw_x_test = [line.split("\t")[1].strip() for line in test_lines]
        raw_y_test = [line.split("\t")[0].strip() for line in test_lines]

    with open('./data/raw/val.txt', 'r', encoding='utf-8') as val_file:
        val_lines = val_file.readlines()
        raw_x_val = [line.split("\t")[1].strip() for line in val_lines]
        raw_y_val = [line.split("\t")[0].strip() for line in val_lines]

    raw_x = [raw_x_train, raw_x_test, raw_x_val]
    raw_y = [raw_y_train, raw_y_test, raw_y_val]

    return raw_x, raw_y


def _tokenize_data(raw_x):
    """
    Tokenizing the data for training.
    """
    assert len(raw_x) == 3
    raw_x_train, raw_x_test, raw_x_val = raw_x

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)

    char_index = tokenizer.word_index
    sequence_length = 200

    x_train = pad_sequences(tokenizer.texts_to_sequences(
        raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(
        raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(
        raw_x_test), maxlen=sequence_length)

    return x_train, x_val, x_test, char_index


def _encode_data(raw_y):
    """
    Encoding the data for training.
    """
    assert len(raw_y) == 3
    raw_y_train, raw_y_test, raw_y_val = raw_y

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    return y_train, y_val, y_test


def _preprocess(raw_x, raw_y):
    """
    Preprocessing the data for training.
    """
    print('\n################### Preprocessing Data ###################\n')
    x_train, x_val, x_test, char_index = _tokenize_data(raw_x)
    y_train, y_val, y_test = _encode_data(raw_y)

    # Dumping the data for training
    dump(x_train, 'data/preprocessed/preprocessed_x_train.joblib')
    dump(x_val, 'data/preprocessed/preprocessed_x_val.joblib')
    dump(x_test, 'data/preprocessed/preprocessed_x_test.joblib')
    dump(char_index, 'data/preprocessed/char_index.joblib')

    dump(y_train, 'data/preprocessed/preprocessed_y_train.joblib')
    dump(y_val, 'data/preprocessed/preprocessed_y_val.joblib')
    dump(y_test, 'data/preprocessed/preprocessed_y_test.joblib')


def main():
    """
    The main method that runs the data preprocessing.
    """
    raw_x, raw_y = _load_data()
    _preprocess(raw_x, raw_y)


if __name__ == '__main__':
    main()
