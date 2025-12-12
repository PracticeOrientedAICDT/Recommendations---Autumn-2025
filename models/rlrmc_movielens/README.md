# RLRMC on MovieLens 1M Dataset

## Dataset

Download the MovieLens 1M Dataset from [here](https://grouplens.org/datasets/movielens/1m/) and store this in a folder called `ml-1m` under the main `dataset` folder, i.e. path should be `dataset/ml-1m`.

## Types of recommendation system implemented in train_rlrmc_movielens.py

- **Riemannian Low-rank Matrix Completion (RLRMC)**: A matrix completion algorithm that uses Riemannian optimization to find a low-rank completion of the user-item rating matrix. It formulates the problem as an optimization on the manifold of fixed-rank matrices.

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
   python models/rlrmc_movielens/train_rlrmc_movielens.py
   ```

## Inference

You can generate recommendations for a specific user using the trained model.

### How it works

The inference script (`inference_rlrmc_movielens.py`) loads the trained model matrices and computes predictions.

1.  **Loads Model**: It loads the user (`L`) and item (`R`) matrices from the saved `.npy` files and the mapping dictionaries from `mappings.json`.
2.  **Scores Candidates**: It calculates the predicted rating for all items for the target user by multiplying the corresponding user and item factors (completion of the matrix).
3.  **Ranks Items**: It sorts the items by predicted rating and selects the top-K.
4.  **Enriches Data**: It adds movie metadata (titles, genres).
5.  **Outputs Results**: Prints recommendations to the console and optionally saves them to a JSON file.

### Running Inference

To generate recommendations for a user (e.g., User ID 123):

```bash
python models/rlrmc_movielens/inference_rlrmc_movielens.py --user-id 123 --top-k 10
```

### Output Format

The output is a JSON object containing the user ID and a list of recommendations:

```json
{
  "user_id": "123",
  "top_k": 10,
  "recommendations": [
    {
      "itemID": "260",
      "title": "Star Wars: Episode IV - A New Hope (1977)",
      "genres": "Action|Adventure|Fantasy|Sci-Fi",
      "prediction": 4.85
    },
    ...
  ]
}
```
