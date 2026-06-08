import pandas as pd
from surprise import SVD, KNNBasic, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pickle
import os

# ── LOAD DATA ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('processed_data/netflix_parsed.csv')
print(f"Loaded: {df.shape}")

# ── PREPARE DATA FOR SURPRISE ──────────────────────────────
# Surprise needs only 3 columns: user_id, movie_id, rating
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    df[['user_id', 'movie_id', 'rating']], 
    reader
)

# ── TRAIN TEST SPLIT ───────────────────────────────────────
print("\nSplitting data...")
trainset, testset = train_test_split(data, 
                                     test_size=0.2, 
                                     random_state=42)

print(f"Training samples: {trainset.n_ratings:,}")
print(f"Test samples:     {len(testset):,}")


# ══════════════════════════════════════════════════════════
# MODEL 1 — SVD
# ══════════════════════════════════════════════════════════
print("\n" + "="*50)
print("Training MODEL 1: SVD")
print("="*50)

svd_model = SVD(
    n_factors=50,      # 50 hidden taste factors
    n_epochs=20,       # 20 training rounds
    lr_all=0.005,      # learning rate
    reg_all=0.02,      # regularization
    random_state=42
)

svd_model.fit(trainset)
print("✅ SVD Training done!")

# Evaluate SVD
svd_preds = svd_model.test(testset)
svd_rmse  = accuracy.rmse(svd_preds)
svd_mae   = accuracy.mae(svd_preds)
print(f"SVD RMSE: {svd_rmse:.4f}")
print(f"SVD MAE:  {svd_mae:.4f}")


# ══════════════════════════════════════════════════════════
# MODEL 2 — Item Based Collaborative Filtering
# ══════════════════════════════════════════════════════════
print("\n" + "="*50)
print("Training MODEL 2: Item-Based CF")
print("="*50)

item_cf = KNNBasic(
    k=40,
    sim_options={
        'name': 'cosine',
        'user_based': False    # Item based
    },
    verbose=False
)

item_cf.fit(trainset)
print("✅ Item-CF Training done!")

# Evaluate Item-CF
cf_preds = item_cf.test(testset)
cf_rmse  = accuracy.rmse(cf_preds)
cf_mae   = accuracy.mae(cf_preds)
print(f"Item-CF RMSE: {cf_rmse:.4f}")
print(f"Item-CF MAE:  {cf_mae:.4f}")


# ══════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

results = pd.DataFrame({
    'Model':  ['SVD', 'Item-Based CF'],
    'RMSE':   [round(svd_rmse, 4), round(cf_rmse, 4)],
    'MAE':    [round(svd_mae,  4), round(cf_mae,  4)],
})

print(results.to_string(index=False))
winner = 'SVD' if svd_rmse < cf_rmse else 'Item-Based CF'
print(f"\n🏆 Better Model (lower RMSE): {winner}")


# ══════════════════════════════════════════════════════════
# SAVE MODELS
# ══════════════════════════════════════════════════════════
os.makedirs('models', exist_ok=True)

with open('models/svd_model.pkl', 'wb') as f:
    pickle.dump(svd_model, f)

with open('models/item_cf_model.pkl', 'wb') as f:
    pickle.dump(item_cf, f)

print("\n✅ Models saved to models/ folder")