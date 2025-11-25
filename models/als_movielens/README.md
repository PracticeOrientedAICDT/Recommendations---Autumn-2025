# ALS Matrix Factorisation on MovieLens 1M Dataset

## Dataset

Download the MovieLens 1M Dataset from [here](https://grouplens.org/datasets/movielens/1m/) and store this in a folder called `ml-1m` under the main `dataset` folder, i.e. path should be `dataset/ml-1m`.

## Types of recommendation system implemented in train_als_movielens.py

- **Alternating Least Squares (ALS)**: A matrix factorization algorithm that decomposes the user-item interaction matrix into two lower-rank matrices representing user and item factors. This implementation uses PySpark for scalable distributed processing.

## Prerequisites

- A virtual environment using python > v3.13
- Java 8+ (for PySpark)

## Setup

1. In the base directory of the repo:
2. Create and activate a venv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -e .
   pip install pyspark wandb
   ```
4. Run the training script using the venv:
   ```bash
   python models/als_movielens/train_als_movielens.py
   ```

## Inference

You can generate recommendations for a specific user using the trained model.

### How it works

The inference script (`inference_als_movielens.py`) loads the saved Spark ALS model and generates top-N movie recommendations for a given user ID.

1.  **Loads Model**: It loads the persisted Spark ALS model from the `models/als_movielens/spark_model` directory.
2.  **Generates Recommendations**: It uses the `recommendForUserSubset` method of the ALS model to find the best items for the specified user.
3.  **Enriches Data**: It looks up movie titles and genres from the `item_lookup_*.parquet` file created during training.
4.  **Outputs Results**: Displays the recommendations in the console and saves them to a JSON file (e.g., `recommendations_user_123.json`).

### Running Inference

To generate recommendations for a user (e.g., User ID 123):

```bash
python models/als_movielens/inference_als_movielens.py --user-id 123 --top-k 10
```

### Output Format

The script saves the recommendations as a JSON array of objects. Each object contains:

```json
[
  {
    "UserId": 123,
    "MovieId": 260,
    "title": "Star Wars: Episode IV - A New Hope (1977)",
    "genres": "Action|Adventure|Fantasy|Sci-Fi",
    "score": 4.95
  },
  ...
]
```
