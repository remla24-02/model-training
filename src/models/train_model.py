"""
Train models for predictions.
"""

from pathlib import Path
import sys
from joblib import dump, load

from src.models.define_model import _get_parameters

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


def main(model_name='trained_model', defined_model='defined_model',
         preprocessed_folder='data/preprocessed'):
    """
    Train the defined model.
    """
    model = load(f'models/{defined_model}.joblib')

    x_train = load(f'{preprocessed_folder}/preprocessed_x_train.joblib')
    y_train = load(f'{preprocessed_folder}/preprocessed_y_train.joblib')
    x_val = load(f'{preprocessed_folder}/preprocessed_x_val.joblib')
    y_val = load(f'{preprocessed_folder}/preprocessed_y_val.joblib')

    params = _get_parameters()

    model.compile(loss=params['loss_function'],
                  optimizer=params['optimizer'], metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=params['batch_train'],
              epochs=params['epoch'], shuffle=True, validation_data=(x_val, y_val))

    # Store trained model
    dump(model, f'models/{model_name}.joblib')


if __name__ == "__main__":
    main()
