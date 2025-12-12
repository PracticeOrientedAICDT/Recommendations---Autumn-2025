# Neural Collaborative Filtering (NCF) on MovieLens 1M Dataset

## Dataset

Download the MovieLens 1M Dataset from [here](https://grouplens.org/datasets/movielens/1m/) and store this in a folder called `ml-1m` under the main `dataset` folder, i.e. path should be `dataset/ml-1m`.

## Types of recommendation system implemented in train_ncf_movielens.py

- **Neural Collaborative Filtering (NCF)**: A deep learning-based framework that replaces the inner product in matrix factorization with a neural architecture. It combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) to model user-item interactions. This implementation uses TensorFlow.

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
   pip install tensorflow wandb
   ```
4. Run the training script using the venv:
   ```bash
   python models/ncf_movielens/train_ncf_movielens.py
   ```

## Inference

You can generate recommendations for a specific user using the trained model.

### How it works

The inference script (`inference_ncf_movielens.py`) uses the trained TensorFlow model to predict scores for candidate items.

1.  **Loads Model**: It rebuilds the NCF model using saved hyperparameters (`metadata.json`) and restores weights from the TensorFlow checkpoint (`checkpoint/`).
2.  **Identifies Candidates**: It selects all items that the user has *not* interacted with yet (based on `train_interactions.json`).
3.  **Predicts Scores**: It runs the model to predict the probability of interaction for all candidate items for the user.
4.  **Enriches Data**: It adds movie metadata (titles, genres) to the results.
5.  **Outputs Results**: Saves the top-N recommendations to a JSON file (e.g., `recommendations_user_123.json`).

### Running Inference

To generate recommendations for a user (e.g., User ID 123):

```bash
python models/ncf_movielens/inference_ncf_movielens.py --user-id 123 --top-k 10
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
    "score": 0.98
  },
  ...
]
```
