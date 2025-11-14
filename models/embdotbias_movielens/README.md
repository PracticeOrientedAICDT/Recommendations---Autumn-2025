# EmbeddingDotBias

**How it works:**
A simple but effective collaborative filtering model that learns user and item embeddings, plus bias terms. Predictions are computed as: user_embedding · item_embedding + user_bias + item_bias + global_bias.

**Key features:**
- Simple architecture (just embeddings + biases)
- Fast training and inference
- Easy to interpret
- Uses PyTorch

**Training:** Learns embeddings and biases through gradient descent to minimize prediction error
**Inference:** Dot product of embeddings plus bias terms gives rating prediction

