# Matrix Factorisation Models

This folder contains various Matrix Factorisation models, which are part of the model zoo for the [Recommendations](https://github.com/PracticeOrientedAICDT/Recommendations---Autumn-2025) project. There are several different versions implemented on different datasets. Eventually a prefered model and method will be selected.

## Matrix Factorisation on IMDb Users' Rating Dataset

### Prerequisites
* A virtual environment using python v3.11.0

### Setup
1) Download the IMDb Users' Ratings Dataset from [here](https://ieee-dataport.org/open-access/imdb-users-ratings-dataset) and store this in a folder called `imdb-user-data` under the `Dataset` folder
2) Install the required dependencies using `pip install -r requirements_imdb.txt`

### Usage
1) Navigate to the `MF_recommender_imdb.ipynb` notebook and run the desired cells.

### Augmenting data
You can augment the IMDb dataset with movie titles using `augment_imdb_data.py`, to do so:
* Run the command `python Dataset/augment_imdb_data.py` from within your virtual environment

This is not required for running the MF_recommender_IMDB.ipynb notebook, but would be required in order to make this model more functional - currently the user must enter a IMDB ID to use the system.

## Matrix Factorisation on MovieLens 1M Dataset 

### Dataset
Download the MovieLens 1M Dataset from [here](https://grouplens.org/datasets/movielens/1m/) and store this in a folder called `ml1m` under the `Dataset` folder

Types implemented:
1) Matrix Factorisation for similarity slates (because you watched [x] you might like ...)
    * `MF_recommender_ml1m.ipynb`
2) RecBole TBC

### 2 - RecBole

Setup:
1) Create venv
2) `pip install torch recbole`
3) Convert dataset to atomic files [guide](https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/MovieLens.md):

    1) `git clone https://github.com/RUCAIBox/RecDatasets`
    2) `cd RecDatasets/conversion_tools`
    3) `pip install -r requirements.txt` (this probably won't install anything as the requirements will be satisfied)
    4) `python run.py --dataset ml-1m --input_path ../../Dataset/ml-1m --output_path ../../Dataset/ml-1m/atomic/ --convert_inter --convert_item --convert_user`
    5) Cry because this doesn't work
    6) Download the files directly from [here](https://drive.google.com/drive/folders/1OkDVEqetvOrtbuWebxl4y1JlZ_YjjfWj)
4) Try to run recbole-debais
5) Cry because that doesn't work either - can't un derstand where to put data/pass data to algorithm, might have to steal code and build own functionality to use :(