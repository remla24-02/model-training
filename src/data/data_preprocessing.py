"""
Preprocess the data to be trained by the learning algorithm.
"""

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import dump


def _load_data():
    """
    Loading the dataset.
    """
    train = [line.strip() for line in open("./data/train.txt", "r").readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    test = [line.strip() for line in open("./data/test.txt", "r").readlines()]
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    val = [line.strip() for line in open("./data/val.txt", "r").readlines()]
    raw_x_val = [line.split("\t")[1] for line in val]
    raw_y_val = [line.split("\t")[0] for line in val]

    return raw_x_train, raw_y_train, raw_x_test, raw_y_test, raw_x_val, raw_y_val


def _tokenize_data(raw_x_train, raw_x_test, raw_x_val):
    """
    Tokenizing the data for training.
    """
    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)

    char_index = tokenizer.word_index
    sequence_length=200

    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

    return x_train, x_val, x_test, char_index


def _encode_data(raw_y_train, raw_y_test, raw_y_val):
    """
    Encoding the data for training.
    """
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    return y_train, y_val, y_test


def _preprocess(raw_x_train, raw_y_train, raw_x_test, raw_y_test, raw_x_val, raw_y_val):
    """
    Preprocessing the data for training.
    """
    print('\n################### Preprocessing Data ###################\n')
    x_train, x_val, x_test, char_index = _tokenize_data(raw_x_train, raw_x_test, raw_x_val)
    y_train, y_val, y_test = _encode_data(raw_y_train, raw_y_test, raw_y_val)

    # Dumping the data for training
    dump(x_train, 'output/preprocessed_x_train.joblib')
    dump(x_val, 'output/preprocessed_x_val.joblib')
    dump(x_test, 'output/preprocessed_x_test.joblib')
    dump(char_index, 'output/char_index.joblib')

    dump(y_train, 'output/preprocessed_y_train.joblib')
    dump(y_val, 'output/preprocessed_y_val.joblib')
    dump(y_test, 'output/preprocessed_y_test.joblib')


def main():
    raw_x_train, raw_y_train, raw_x_test, raw_y_test, raw_x_val, raw_y_val = _load_data()
    _preprocess(raw_x_train, raw_y_train, raw_x_test, raw_y_test, raw_x_val, raw_y_val)


if __name__ == "__main__":
    main()
