import pandas as pd
import pickle
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

# ── LOAD DATA ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('processed_data/netflix_parsed.csv')
movies = df[['movie_id', 'title']].drop_duplicates()
print(f"✅ Loaded: {df.shape}")

# ── PREPARE SURPRISE DATA ──────────────────────────────────
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    df[['user_id', 'movie_id', 'rating']],
    reader
)

trainset, testset = train_test_split(data,
                                     test_size=0.2,
                                     random_state=42)

# ── LOAD SAVED MODELS ──────────────────────────────────────
print("\nLoading models...")
with open('models/svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)

with open('models/item_cf_model.pkl', 'rb') as f:
    item_cf = pickle.load(f)

print("✅ Models loaded!")


# ══════════════════════════════════════════════════════════
# STEP 4A — TOP-10 RECOMMENDATIONS FUNCTION
# ══════════════════════════════════════════════════════════

def get_top10(model, user_id, df, movies, n=10):
    # Movies this user already rated
    rated = set(df[df['user_id'] == user_id]['movie_id'].tolist())
    
    # All movies in dataset
    all_movies = df['movie_id'].unique()
    
    # Predict rating for every UNRATED movie
    preds = []
    for mid in all_movies:
        if mid not in rated:
            est = model.predict(user_id, mid).est
            preds.append((mid, est))
    
    # Sort by predicted rating — highest first
    preds.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 10
    top10 = preds[:n]
    
    # Add movie titles
    results = []
    for mid, score in top10:
        title = movies[movies['movie_id'] == mid]['title'].values
        title = title[0] if len(title) > 0 else 'Unknown'
        results.append({
            'movie_id': mid,
            'title': title,
            'predicted_rating': round(score, 2)
        })
    
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════
# STEP 4B — SHOW SAMPLE RECOMMENDATIONS
# ══════════════════════════════════════════════════════════

# Pick 3 random users
sample_users = df['user_id'].sample(3, random_state=42).tolist()

print("\n" + "="*55)
print("SAMPLE RECOMMENDATIONS — SVD MODEL")
print("="*55)

for uid in sample_users:
    # Movies this user actually rated highly
    liked = df[(df['user_id'] == uid) & 
               (df['rating'] >= 4)]['title'].tolist()[:3]
    
    print(f"\n👤 User {uid}")
    print(f"   Liked: {liked}")
    print(f"   Top 10 Recommendations:")
    
    recs = get_top10(svd_model, uid, df, movies)
    for i, row in recs.iterrows():
        print(f"   {i+1:2}. {row['title']:<40} "
              f"(predicted: ⭐{row['predicted_rating']})")


# ══════════════════════════════════════════════════════════
# STEP 5 — MAP@10 CALCULATION
# ══════════════════════════════════════════════════════════

def average_precision_at_k(recommended, relevant, k=10):
    recommended = recommended[:k]
    hits  = 0
    score = 0.0
    
    for i, mid in enumerate(recommended):
        if mid in relevant:
            hits  += 1
            score += hits / (i + 1)
    
    if len(relevant) == 0:
        return 0.0
    
    return score / min(k, len(relevant))


def compute_map10(model, df, n_users=200):
    print(f"\nComputing MAP@10 on {n_users} users...")
    
    # Split into train/test by user
    train_df = df.sample(frac=0.8, random_state=42)
    test_df  = df.drop(train_df.index)
    
    # Pick users who have enough test ratings
    valid_users = (test_df[test_df['rating'] >= 3.5]
                   ['user_id'].value_counts())
    valid_users = valid_users[valid_users >= 3].index.tolist()
    
    # Sample n_users
    import random
    random.seed(42)
    selected = random.sample(valid_users, 
                             min(n_users, len(valid_users)))
    
    ap_scores = []
    
    for i, uid in enumerate(selected):
        if i % 50 == 0:
            print(f"  Processing user {i}/{len(selected)}...")
        
        # Relevant = movies rated >= 3.5 in test set
        relevant = set(
            test_df[(test_df['user_id'] == uid) & 
                    (test_df['rating'] >= 3.5)]['movie_id'].tolist()
        )
        
        if not relevant:
            continue
        
        # Movies rated in train (exclude from recommendations)
        train_rated = set(
            train_df[train_df['user_id'] == uid]['movie_id'].tolist()
        )
        
        # Candidate movies (not in train)
        candidates = [m for m in df['movie_id'].unique() 
                      if m not in train_rated]
        
        # Predict and rank
        preds = [(m, model.predict(uid, m).est) 
                 for m in candidates]
        preds.sort(key=lambda x: x[1], reverse=True)
        
        recommended = [m for m, _ in preds[:10]]
        
        ap = average_precision_at_k(recommended, relevant)
        ap_scores.append(ap)
    
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0


# ── COMPUTE MAP@10 FOR BOTH MODELS ────────────────────────
print("\n" + "="*55)
print("EVALUATION — MAP@10")
print("="*55)

map_svd = compute_map10(svd_model, df, n_users=200)
print(f"✅ SVD     MAP@10: {map_svd:.4f}")

map_cf = compute_map10(item_cf, df, n_users=200)
print(f"✅ Item-CF MAP@10: {map_cf:.4f}")


# ══════════════════════════════════════════════════════════
# FINAL RESULTS TABLE
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("FINAL MODEL COMPARISON")
print("="*55)

results = pd.DataFrame({
    'Model':  ['SVD', 'Item-Based CF'],
    'RMSE':   [0.9423, 1.0231],   # paste your values from models.py
    'MAP@10': [round(map_svd, 4), round(map_cf, 4)]
})

print(results.to_string(index=False))

winner_rmse = 'SVD' if 0.9423 < 1.0231 else 'Item-CF'
winner_map  = 'SVD' if map_svd > map_cf else 'Item-CF'

print(f"\n🏆 Better RMSE:   {winner_rmse}")
print(f"🏆 Better MAP@10: {winner_map}")

# Save results
results.to_csv('processed_data/model_results.csv', index=False)
print("\n✅ Results saved to processed_data/model_results.csv")