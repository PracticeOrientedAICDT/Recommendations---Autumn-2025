# Embedding Dot Bias on MovieLens 1M Dataset

## Dataset

Download the MovieLens 1M Dataset from [here](https://grouplens.org/datasets/movielens/1m/) and store this in a folder called `ml-1m` under the main `dataset` folder, i.e. path should be `dataset/ml-1m`.

## Types of recommendation system implemented in train_embdotbias_movielens.py

- **Embedding Dot Bias**: A collaborative filtering model that learns user and item embeddings along with bias terms. It predicts ratings by computing the dot product of user and item embeddings plus the respective bias terms. This implementation uses PyTorch.

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
   pip install torch wandb
   ```
4. Run the training script using the venv:
   ```bash
   python models/embdotbias_movielens/train_embdotbias_movielens.py
   ```

## Inference

You can generate recommendations for a specific user using the trained model.

### How it works

The inference script (`inference_embdotbias_movielens.py`) loads the trained PyTorch model and calculates scores for all items to find the best matches for a user.

1.  **Loads Model**: It reconstructs the `EmbeddingDotBias` model architecture using metadata and loads the trained weights (`model.pth`) and class mappings (`classes.json`).
2.  **Scores Candidates**: It computes the predicted score (dot product of embeddings + bias) for every item in the dataset for the target user.
3.  **Ranks Items**: It sorts items by score and selects the top-K.
4.  **Enriches Data**: It merges the results with movie titles and genres from the dataset.
5.  **Outputs Results**: Prints recommendations to the console and optionally saves them to a JSON file.

### Running Inference

To generate recommendations for a user (e.g., User ID 123):

```bash
python models/embdotbias_movielens/inference_embdotbias_movielens.py --user-id 123 --top-k 10
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
