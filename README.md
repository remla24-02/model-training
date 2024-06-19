![Pylint Score](https://img.shields.io/badge/pylint-0-brightgreen)

![accuracy](https://img.shields.io/badge/accuracy-0.0-blue)
![precision](https://img.shields.io/badge/precision-0.0-blue)
![recall](https://img.shields.io/badge/recall-0.0-blue)
![f1](https://img.shields.io/badge/f1-0.0-blue)
![roc_auc](https://img.shields.io/badge/roc_auc-0.0-blue)


# REMLA Team 2 Model
This is Team 2's repository for Assignment A1 for Release Engineering for Machine Learning Applications 2023/24.  
It contains a DVC pipeline with public download access to a AWS S3 Bucket.  
The metrics displayed in the badges above can be found back in the evaluation folder (specifically [here](https://github.com/remla24-02/model-training/blob/main/evaluation_results/metrics.json)).

## Prerequisites 
Python 3.12 is required to run the code.  
To install the required packages a pyproject.toml has been provided for use with [Poetry](https://python-poetry.org/docs/).
This has been testing using Poetry version 1.7.1.
Poetry can be installed with pip with the following command:
``` console
pip install poetry
```

## Installation

``` console
git clone https://github.com/remla24-02/model-training.git
cd model
poetry install --no-root
```

This cloned the repository and installed all the packages into an environment.
Next, open a new shell for the environment with the following command:
``` console
poetry shell
```

The data to be used during the pipeline is automatically downloaded for the remote Bucket during the pipeline.
To run the pipeline, do:
``` console
dvc repro
```

If you wish to view the data prior to running the pipeline:

``` console
poetry run python3 src/data/get_data.py
```

This will save the test, train and validation data to the `data/raw` folder under root.

To download a pretrained model.

``` console
poetry run python3 src/models/get_model.py
```

This will download a pretrained model to the `models` folder under root.

Next, for the code quality Pylint is used which can be run in the poetry shell as:

``` console
pylint src
```

This will show all code smells and provide a score for the codebase

Lastly, all the ML tests can be run with:
(note that this messes with saved data and models)

``` console
pytest tests
```

### Project group specifics
To push data to the DVC remote you need the access key id and secret and add those to the `.dvc` folder in a file called `config.local`.

``` text
['remote "aws_s3"']
    access_key_id = <ID>
    secret_access_key = <SECRET>
```

## Project structure

```console
$ tree
.
├── data
│   ├── preprocessed            # <-- Directory with processed data
│   └── raw                     # <-- Directory with raw data
├── docs
│   └── ACTIVITY.md             # <-- Activity tracking per group member
├── dvc.lock
├── dvc.yaml                    # <-- DVC pipeline
├── evaluation                  
│   ├── metrics.json            # <-- Final metrics (e.g. accuracy)
│   └── plots                   # <-- Data points for e.g. ROC
│       └── sklearn
│           ├── cm.json
│           ├── prc.json
│           └── roc.json
├── LICENSE
├── models                      # <-- Directory for model files
├── notebooks
├── poetry.lock
├── pyproject.toml              # <-- File for dependencies (Poetry)
├── README.md
├── requirements.txt
└── src                         # <-- Source code for the pipeline
    ├── data
    │   ├── data_preprocessing.py
    │   └── get_data.py
    └── models
        ├── define_model.py     # <-- Creates the model
        ├── get_model.py        # <-- Download model from Bucket
        ├── evaluate_model.py    # <-- Evaluates the model
        └── train_model.py      # <-- Trains the model
```

## Pipeline stages

- `get_data`: Downloads the raw data from remote AWS S3 Bucket
- `preprocess`: Preprocessed the data for training. Outputs tokenized and encoded data files in `data/preprocessed` in the root directory.
- `define_model`: Creates the untrained model and stores the file in `models` in the root directory.
- `train_model`: Trains the defined model and stores the file in `models` in the root directory. 
- `evaluate_model`: Evaluates the models performance and saves the metrics in the `evaluation` folder.

## Plots
To get interactive plots you can run:

``` console
dvc plots show
```

### Confusion Matric
![ROC Curve](https://raw.githubusercontent.com/remla24-02/model-training/main/evaluation_results/plots/cm.png)

### ROC Curve
![ROC Curve](https://raw.githubusercontent.com/remla24-02/model-training/main/evaluation_results/plots/roc.png)

### Precision-Recall
![ROC Curve](https://raw.githubusercontent.com/remla24-02/model-training/main/evaluation_results/plots/prc.png)