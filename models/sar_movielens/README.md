# SAR on MovieLens 1M Dataset

## Dataset

Download the MovieLens 1M Dataset from [here](https://grouplens.org/datasets/movielens/1m/) and store this in a folder called `ml-1m` under the main `dataset` folder, i.e. path should be `dataset/ml-1m`.

## Types of recommendation system implemented in train_sar_movielens.py

- **Simple Algorithm for Recommendation (SAR)**: A fast and scalable algorithm for personalized recommendations based on user transaction history. It creates a similarity matrix between items based on co-occurrences and recommends similar items to those a user has interacted with.

## Prerequisites

- A virtual environment using python > v3.13

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
   pip install wandb
   ```
4. Run the training script using the venv:
   ```bash
   python models/sar_movielens/train_sar_movielens.py
   ```

## Inference

You can generate recommendations for a specific user using the trained model.

### How it works

The inference script (`inference_sar_movielens.py`) loads the trained SAR model and generates recommendations.

1.  **Loads Model**: It loads the persisted `SAR` model object from `model.joblib`.
2.  **Generates Recommendations**: It calls the `recommend_k_items` method of the model for the specified user, which computes scores based on the item similarity matrix and the user's history.
3.  **Enriches Data**: It looks up movie titles and genres if metadata is available.
4.  **Outputs Results**: Saves the top-N recommendations to a JSON file.

### Running Inference

To generate recommendations for a user (e.g., User ID 123):

```bash
python models/sar_movielens/inference_sar_movielens.py --user-id 123 --top-k 10
```

### Output Format

The script saves the recommendations as a JSON array of objects. Each object contains:

```json
[
  {
    "userID": 123,
    "itemID": 260,
    "title": "Star Wars: Episode IV - A New Hope (1977)",
    "genres": "Action|Adventure|Fantasy|Sci-Fi",
    "prediction": 12.5
  },
  ...
]
```
