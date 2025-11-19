# SASRec on MovieLens

Self-Attentive Sequential Recommendation (SASRec) is a sequential recommendation model that uses a Transformer-based architecture to capture the long-term semantics and short-term dependencies of user interactions. It models the user's sequence of item interactions to predict the next item they are likely to interact with.

## Training

Train the model using the provided script:
```bash
python examples/00_quick_start/scripts/train_sasrec_movielens.py --data-size 1m --epochs 100
```

## Inference

Generate recommendations for a specific user:
```bash
python examples/00_quick_start/scripts/inference_sasrec_movielens.py --user-id 1 --top-k 10
```

## Implementation Details

- **Dataset**: MovieLens (1M default)
- **Architecture**: Transformer Encoder (Multi-Head Attention, Point-wise Feed-Forward)
- **Objective**: Next-item prediction (Binary Cross-Entropy with Negative Sampling)
- **Metrics**: NDCG@10, Hit Rate@10

