# RLRMC (Riemannian Low-rank Matrix Completion)

**How it works:**
RLRMC performs matrix completion on a Riemannian manifold, which respects geometric constraints of the data. It learns low-rank matrices L and R such that L×R approximates the user-item rating matrix, but does so in a way that respects the manifold structure.

**Key features:**
- Geometric approach to matrix factorization
- Handles missing data naturally
- Good theoretical properties
- Works well with sparse ratings

**Training:** Optimizes L and R matrices on Riemannian manifold using gradient descent
**Inference:** Computes L[user] × R[item] to get rating predictions

