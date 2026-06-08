import pandas as pd
import os

def parse_netflix_data(filepath):
    data = []
    current_movie = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(':'):
                current_movie = int(line[:-1])
            else:
                parts = line.split(',')
                user_id = int(parts[0])
                rating  = int(parts[1])
                date    = parts[2]
                data.append([current_movie, user_id, rating, date])
    
    df = pd.DataFrame(data, columns=['movie_id', 'user_id', 'rating', 'date'])
    return df


# ── STEP 1: Load rating files ─────────────────────────────

print("Loading file 1...")
df1 = parse_netflix_data('raw_data/combined_data_1.txt')

# If laptop is slow, just use file 1 and skip 2,3,4
# df = df1.sample(n=1_000_000, random_state=42)

print("Loading file 2...")
df2 = parse_netflix_data('raw_data/combined_data_2.txt')

print("Loading file 3...")
df3 = parse_netflix_data('raw_data/combined_data_3.txt')

print("Loading file 4...")
df4 = parse_netflix_data('raw_data/combined_data_4.txt')


# ── STEP 2: Combine all files ─────────────────────────────

df_all = pd.concat([df1, df2, df3, df4], ignore_index=True)
print(f"Total ratings: {df_all.shape}")

# Sample 2 million rows
df = df_all.sample(n=2_000_000, random_state=42)
print(f"Sampled: {df.shape}")


# ── STEP 3: Load movie titles (safe version) ──────────────

print("Loading movie titles...")

rows = []
with open('raw_data/movie_titles.csv', 'r', encoding='latin-1') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',', 2)
        if len(parts) == 3:
            try:
                movie_id = int(parts[0])
                year     = parts[1]
                title    = parts[2]
                rows.append([movie_id, year, title])
            except ValueError:
                continue   # Skip any weird lines

movies = pd.DataFrame(rows, columns=['movie_id', 'year', 'title'])

print(f"✅ Movies loaded: {movies.shape}")
print(movies.head(10))


# ── STEP 4: Merge ─────────────────────────────────────────

df_full = df.merge(movies, on='movie_id', how='left')
print(f"Merged shape: {df_full.shape}")
print(df_full.head())


# ── STEP 5: Basic checks ──────────────────────────────────

print("\nMissing values:")
print(df_full.isnull().sum())

print("\nRating range:", df_full['rating'].min(), "to", df_full['rating'].max())


# ── STEP 6: Save ──────────────────────────────────────────

os.makedirs('processed_data', exist_ok=True)
df_full.to_csv('processed_data/netflix_parsed.csv', index=False)
print("\n✅ Data saved to processed_data/netflix_parsed.csv")