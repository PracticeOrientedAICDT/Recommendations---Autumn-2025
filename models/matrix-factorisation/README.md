# Matrix Factorisation Models

This folder contains the Matrix Factorisation models, which are part of the model zoo for the [Recommendations](https://github.com/PracticeOrientedAICDT/Recommendations---Autumn-2025) project.

## Prerequisites

## Setup
1) Download the IMDb Users' Ratings Dataset from [here](https://ieee-dataport.org/open-access/imdb-users-ratings-dataset) and store this in a folder called `imdb-user-data` under the `Dataset` folder
2) Create and activate a virtual environment using python v3.11.0:
    - TODO - expand on how to do this
3) Install the required dependencies using `pip install -r requirements.txt`
4) Augment the IMDb dataset with movie titles using `augment_imdb_data.py`
    - Run the command `python Dataset/augment_imdb_data.py` from within your virtual environment

## Usage
1) Navigate to the `MF_recommender.ipynb` notebook and run the desired cells.
