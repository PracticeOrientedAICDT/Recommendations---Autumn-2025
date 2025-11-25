# Matrix Factorisation Models

This folder contains various Matrix Factorisation models, which are part of the model zoo for the [Recommendations](https://github.com/PracticeOrientedAICDT/Recommendations---Autumn-2025) project. There are several different versions implemented on different datasets. Eventually a prefered model and method will be selected.

## Matrix Factorisation on MovieLens 1M Dataset 

### Dataset
Download the MovieLens 1M Dataset from [here](https://grouplens.org/datasets/movielens/1m/) and store this in a folder called `ml1m` under the main `Dataset` folder, i.e. path should be `Dataset/ml-1m`.

Types of reccommendation system implemented in `MF_recommender_ml1m.ipynb` (see main folder)
1) Matrix Factorisation for similarity slates (because you watched [x] you might like ...)
2) Matrix Factorisation for Top-N for specific user

### Prerequisites
* A virtual environment using python > v3.13

### Setup:
1) In the base directory of the repo (same folder as `MF_recommender_ml1m.ipynb`; NOT this one)
2) Create and activate a venv
3) Install the required dependencies using `pip install -r models/matrix_factorisation/requirements_nmf_mf_ml1m.txt`
4) Open `MF_recommender_ml1m.ipynb`and run all cells using the venv as the kernel.


## Matrix Factorisation on IMDb Users' Rating Dataset (DEPRECATED)

### Prerequisites
* A virtual environment using python v3.11.0

### Setup
1) Download the IMDb Users' Ratings Dataset from [here](https://ieee-dataport.org/open-access/imdb-users-ratings-dataset) and store this in a folder called `imdb-user-data` under a `Dataset` folder *within* this folder i.e. path should be `models/matrix_factorisation/Dataset/imdb-user-data`.
2) Install the required dependencies using `pip install -r requirements_imdb.txt`

### Usage
1) Open the `MF_recommender_imdb.ipynb` notebook and run the desired cells.

### Augmenting data
You can augment the IMDb dataset with movie titles using `augment_imdb_data.py`, to do so:
* Run the command `python Dataset/augment_imdb_data.py` from within your virtual environment

This is not required for running the MF_recommender_IMDB.ipynb notebook, but would be required in order to make this model more functional - currently the user must enter a IMDB ID to use the system.