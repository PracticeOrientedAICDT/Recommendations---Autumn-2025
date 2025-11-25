# SASRec on MovieLens 1M Dataset

## Dataset

Download the MovieLens 1M Dataset from [here](https://grouplens.org/datasets/movielens/1m/) and store this in a folder called `ml-1m` under the main `dataset` folder, i.e. path should be `dataset/ml-1m`.

## Types of recommendation system implemented in train_sasrec_movielens.py

- **Self-Attentive Sequential Recommendation (SASRec)**: A sequential recommendation model that uses a self-attention mechanism (Transformer architecture) to capture long-term semantics and short-term dependencies in user action sequences to predict the next item.

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
   python models/sasrec_movielens/train_sasrec_movielens.py
   ```

## Inference

You can generate recommendations for a specific user using the trained model.

### How it works

The inference script (`inference_sasrec_movielens.py`) predicts the next item a user is likely to interact with based on their history.

1.  **Loads Model and Data**: It loads the model configuration, user history, item metadata, and mappings. It reconstructs the SASRec model and loads the trained weights.
2.  **Prepares Input**: It retrieves the user's interaction history and creates a sequence input for the model.
3.  **Predicts Scores**: It passes the sequence to the model to get logits (scores) for all items (candidates) as the next probable item.
4.  **Ranks Items**: It sorts the items by score and selects the top-K.
5.  **Outputs Results**: Prints recommendations to the console and outputs them as a JSON array.

### Running Inference

To generate recommendations for a user (e.g., User ID 123):

```bash
python models/sasrec_movielens/inference_sasrec_movielens.py --user-id 123 --top-k 10
```

### Output Format

The output is a JSON array of objects:

```json
[
  {
    "userID": 123,
    "itemID": 260,
    "title": "Star Wars: Episode IV - A New Hope (1977)",
    "genres": "Action|Adventure|Fantasy|Sci-Fi",
    "score": 8.45
  },
  ...
]
```
