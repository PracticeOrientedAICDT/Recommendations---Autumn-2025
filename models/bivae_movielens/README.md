# BiVAE on MovieLens 1M Dataset

## Dataset

Download the MovieLens 1M Dataset from [here](https://grouplens.org/datasets/movielens/1m/) and store this in a folder called `ml-1m` under the main `dataset` folder, i.e. path should be `dataset/ml-1m`.

## Types of recommendation system implemented in train_bivae_movielens.py

- **Bilateral Variational Autoencoder (BiVAE)**: A generative model for collaborative filtering that extends the standard Variational Autoencoder (VAE). It models user-item interactions by learning bilateral latent representations for both users and items, optimizing a lower bound on the log-likelihood of the interaction data. This implementation uses the Cornac framework (PyTorch backend).

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
   pip install cornac torch wandb
   ```
4. Run the training script (see below).

## How it works

### Training (`train_bivae_movielens.py`)
The training script performs the following steps:
1.  **Data Loading**: Loads the MovieLens 1M dataset.
2.  **Preprocessing**: Converts User and Item IDs to strings and splits the data into training and test sets (default 75/25 split).
3.  **Cornac Dataset**: Converts the dataframe into a Cornac `Dataset` object, which handles indexing and batching.
4.  **Model Initialization**: Initializes the `BiVAECF` model with configurable hyperparameters (latent dimension, encoder structure, learning rate, etc.).
5.  **Training**: Fits the model to the training data using the specified number of epochs.
6.  **Saving**: Saves the trained model artifacts and user/item ID mappings to the `models/bivae_movielens` directory.
7.  **Evaluation**: Evaluates the model on the test set using ranking metrics (MAP, NDCG, Precision, Recall) and logs results to Weights & Biases (if enabled).

### Inference (`inference_bivae_movielens.py`)
The inference script generates recommendations for a specific user:
1.  **Load Mappings**: Reads the saved `mappings.json` to map external User/Item IDs to the model's internal indices.
2.  **Load Model**: Loads the trained `BiVAECF` model from the artifact directory.
3.  **Scoring**: Computes scores for all items for the target user using the model's latent representations.
4.  **Ranking**: Selects the top-k items with the highest scores.
5.  **Enrichment**: Maps internal item indices back to original MovieLens IDs and merges with item metadata (title, genres).
6.  **Output**: Returns the recommendations in JSON format.

## Running the Training Script

To train the model with default settings:

```bash
python models/bivae_movielens/train_bivae_movielens.py
```

Common arguments:
- `--epochs`: Number of training epochs (default: 500)
- `--latent-dim`: Size of the latent dimension (default: 50)
- `--batch-size`: Batch size (default: 128)
- `--no-wandb`: Disable Weights & Biases logging

## Running the Inference Script

To generate recommendations for a specific user (e.g., User ID "1"):

```bash
python models/bivae_movielens/inference_bivae_movielens.py --user-id 1
```

Options:
- `--top-k`: Number of recommendations to return (default: 10)
- `--output`: Path to save the output JSON file.

## Output Format

The inference script outputs a JSON object containing the user ID and the list of recommendations:

```json
{
  "user_id": "1",
  "top_k": 10,
  "recommendations": [
    {
      "itemID": "1193",
      "title": "One Flew Over the Cuckoo's Nest (1975)",
      "genres": "Drama",
      "prediction": 0.9823
    },
    {
      "itemID": "661",
      "title": "James and the Giant Peach (1996)",
      "genres": "Animation|Children's|Musical",
      "prediction": 0.9541
    }
    // ...
  ]
}
```
first eval results : 
:          test/map 0.27686
:         test/ndcg 0.41096
: test/precision@10 0.37551
:    test/recall@10 0.14303
