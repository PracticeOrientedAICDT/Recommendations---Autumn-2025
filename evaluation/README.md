
Evaluates recommendations from a CSV file. Calculates precision, recall, NDCG, diversity, etc.

## Quick Start

```bash
python eval_recommendations.py
```

Uses defaults: `recommendations.csv`, `movie_embeddings.pkl`, k=10, saves to `evaluation_results/`

## Input File Format

Your CSV needs these columns:

```csv
user_id,item_id,relevance,predicted_score
1,364,0,0.0066
1,48,1,0.0041
1,2096,0,0.0037
```

- `user_id`: User ID (int)
- `item_id`: Movie/item ID (int)  
- `relevance`: 1 if relevant, 0 if not (int)
- `predicted_score`: Model's prediction score (float, higher = better)

## Usage

```bash
python eval_recommendations.py \
    --input recommendations.csv \
    --embeddings_path movie_embeddings.pkl \
    --k 10 \
    --output_dir evaluation_results
```

## Output

Creates two files in `evaluation_results/`:
- `user_metrics.csv` - metrics per user
- `overall_metrics.json` - aggregated metrics

## Example Workflow

```bash
# 1. Generate recommendations (if you don't have them)
python generate_recommendations.py --output recommendations.csv

# 2. Evaluate
python eval_recommendations.py

# 3. Check results
cat evaluation_results/overall_metrics.json
```

## Notes

- Embeddings file is optional (only needed for diversity metrics)
- If embeddings missing, ILS will be 0.0
- Make sure relevance labels are correct (1 = relevant item in test set)
