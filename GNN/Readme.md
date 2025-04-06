# Movie Recommendation System using LightGCN (GNN-based)

This project implements a **Graph Neural Network (LightGCN)** based recommender system using the **MovieLens** dataset. It leverages implicit feedback (ratings ≥ 4.0), genre-based embeddings, popularity-aware sampling, and includes cold-start strategies for unseen users/items. The model is optimized to run with **limited memory (12GB RAM)** and utilize **GPU acceleration**.

---

## Features

- ✅ Uses **LightGCN** for collaborative filtering
- ✅ Filters ratings ≥ 4.0 for implicit feedback
- ✅ Integrates **multi-hot genre embeddings** for movies
- ✅ Adds **popularity-aware negative sampling**
- ✅ Optimized to run on GPU and minimize RAM usage
- ✅ Handles **cold-start users/items** using genre + popularity fallback
- ✅ Saves trained model for later inference

---

## Dataset

- Uses **MovieLens 1M / 10M / 20M / Full** depending on memory constraints.
- `.dat` files like `ratings.dat` and `movies.dat` are parsed using `pd.read_csv(..., sep='::', engine='python')`

### Movie Metadata
- Loaded from `movies.dat`
- Format: `MovieID::Title::Genres`

### Ratings
- Loaded from `ratings.dat`
- Format: `UserID::MovieID::Rating::Timestamp`

---

## Architecture

### LightGCN Model
- `num_users`, `num_items`, `embed_dim`, `num_layers`
- Optional `genre_feat_matrix` to include genre signal
- Uses **LightGCN propagation** over interaction graph

### Loss Function
- **Bayesian Personalized Ranking (BPR)** loss:
```python
loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
```

### Negative Sampling
- Supports both **uniform** and **popularity-weighted** sampling

---

## Training

```python
train_model(
    model, data,
    epochs=5,
    batch_size=1024,
    lr=0.005,
    use_popularity_sampling=True
)
```

- Trains LightGCN with genre-aware embeddings
- Uses mini-batching to reduce memory load
- Cached `model.dataset` inside model to avoid repeated arguments
- `torch.cuda.empty_cache()` after each epoch to reduce memory pressure

---

## Evaluation

```python
evaluate_model(model, data, K=10)
```

- Computes **Recall@10** and **NDCG@10** on test interactions

---

## Cold Start Handling

### For New Users:
- Recommend most popular movies
- Diversify based on genre

### For New Items:
- Use genre-based embeddings to match with users of similar taste

---

## Saving the Model

### Save full model:
```python
torch.save(model, "lightgcn_model.pt")
```

### Save just state dict:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'num_users': model.num_users,
    'num_items': model.num_items,
    'genre_feat_matrix': model.item_genres_tensor.cpu().numpy(),
}, "lightgcn_checkpoint.pt")
```

---

## Inference Example

```python
user_id = 0
recs = model.recommend(user_id, dataset, top_k=5)
for rank, (item, score) in enumerate(recs, 1):
    print(f"{rank}. {dataset.movie_titles[item]} (score={score:.3f})")
```

---

## Notes
- Make sure to use the right `encoding='latin-1'` for `.dat` files
- Use `engine='python'` with `sep='::'` while loading with `pandas.read_csv`
- Torch Sparse might not install in some environments; fallback to `scipy.sparse` if needed
- If using `.pt` model directly, ensure `torch.load(..., map_location=device)` for GPU compatibility

---

## Example Output
```bash
[Epoch 5] Avg BPR Loss = 0.1777
Recall@10: 0.0000, NDCG@10: 0.0000
Top 5 recommendations for User 0:
1. Lion King, The (1994)
2. Babe (1995)
3. Fantasia (1940)
4. Little Mermaid, The (1989)
5. Shawshank Redemption, The (1994)
```

---


Made with ❤️ for Graph AI.

