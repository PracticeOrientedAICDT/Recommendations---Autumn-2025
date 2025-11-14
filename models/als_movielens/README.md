# ALS (Alternating Least Squares)

**How it works:**
ALS is a matrix factorization method that learns low-dimensional embeddings for users and items. It decomposes the user-item rating matrix into two smaller matrices (user factors × item factors) by alternating between fixing one and optimizing the other.

**Key features:**
- Handles sparse data well
- Can incorporate implicit feedback
- Distributed/scalable (uses Spark)
- Fast inference (just matrix multiplication)

**Training:** Alternates between solving for user embeddings and item embeddings until convergence
**Inference:** Computes dot product between user embedding and all item embeddings

