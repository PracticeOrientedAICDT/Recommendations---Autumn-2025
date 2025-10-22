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
    - Run the command `python Dataset/augment_imdb_data.py` from within your virtual environment

This is not required for running the MF_recommender_IMDB.ipynb notebook, but would be required in order to make this model more functional - currently the user must enter a IMDB ID to use the system.


