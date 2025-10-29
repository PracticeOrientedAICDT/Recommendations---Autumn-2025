# MFvsTransformer_RecSys

A movie recommendation system comparing **Matrix Factorization (MF)** and **TransformerRecSys** models trained on the MovieLens 1M dataset.

## Project Structure
- `train.py` – Train and evaluate models.
- `recommend_MFvsTransformer.py` – Generate top-K movie recommendations for a given user.
- `models_MFvsTransformer.py` – Defines both MF and TransformerRecSys architectures.
- `models/` – Stores trained model checkpoints.
- `data/ml-1m/` – MovieLens dataset files.

## Usage

### Train the models
```bash
python /train.py
