[![DVC-metrics](https://img.shields.io/badge/dynamic/json?style=flat-square&colorA=grey&colorB=F46737&label=Average%20Precision&url=https://github.com/remla24-02/model/raw/main/eval/metrics.json&query=avg_prec.test)](https://github.com/remla24-02/model/raw/main/evaluation/metrics.json)


# REMLA Team 2 Model
This is Team 2's repository for Assignment A1 for Release Engineering for Machine Learning Applications 2023/24.  
It contains a DVC pipeline with public download access to a AWS S3 Bucket.

## Prerequisites 
Python 3.12 is required to run the code.  
To install the required packages a pyproject.toml has been provided for use with [Poetry](https://python-poetry.org/docs/).  This has been testing using Poetry version 1.7.1.

## Installation

``` console
git clone https://github.com/remla24-02/model.git
cd model
poetry install --no-root
```

The data to be used during the pipleline is automatically downloaded for the remote Bucket during the pipeline.
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
        ├── predict_model.py    # <-- Evaluates the model
        └── train_model.py      # <-- Trains the model
```