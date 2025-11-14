# SAR (Simple Algorithm for Recommendation)

**How it works:**
SAR is a fast, scalable collaborative filtering algorithm that uses item co-occurrence patterns. It builds an item-to-item similarity matrix based on how often items are rated together by users, then uses this to recommend items similar to what a user has already liked.

**Key features:**
- Fast training and inference (no iterative optimization)
- Memory efficient (stores similarity matrix)
- Good for implicit feedback (ratings, clicks, views)

**Training:** Builds co-occurrence matrices and computes item similarities
**Inference:** Multiplies user's item history with similarity matrix to get scores

