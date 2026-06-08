import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("Loading data...")
df = pd.read_csv('processed_data/netflix_parsed.csv')

# Fix year column — remove .0
df['year'] = df['year'].fillna(0).astype(int)

# Fix date column — convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract year from date (rating year — different from movie release year)
df['rating_year'] = df['date'].dt.year
df['rating_month'] = df['date'].dt.month

print(f"✅ Loaded: {df.shape}")
print(df.head())
print(df.dtypes)

print("=" * 50)
print("BASIC STATISTICS")
print("=" * 50)

print(f"Total Ratings:    {len(df):,}")
print(f"Unique Users:     {df['user_id'].nunique():,}")
print(f"Unique Movies:    {df['movie_id'].nunique():,}")
print(f"Date Range:       {df['date'].min()} to {df['date'].max()}")
print(f"Avg Rating:       {df['rating'].mean():.2f}")
print(f"Median Rating:    {df['rating'].median():.1f}")


print("\n📊 Plotting Rating Distribution...")

plt.figure(figsize=(8, 5))

counts = df['rating'].value_counts().sort_index()
bars = plt.bar(counts.index, counts.values, 
               color=['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db'],
               edgecolor='black', width=0.6)

# Add count labels on top of bars
for bar, count in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + 10000,
             f'{count:,}', 
             ha='center', fontsize=10)

plt.title('Rating Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Star Rating')
plt.ylabel('Number of Ratings')
plt.xticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.savefig('processed_data/rating_distribution.png')
plt.show()
print("✅ Saved rating_distribution.png")


print("\n📊 Plotting User Activity...")

user_counts = df['user_id'].value_counts()

print(f"Avg ratings per user:    {user_counts.mean():.1f}")
print(f"Median ratings per user: {user_counts.median():.1f}")
print(f"Max ratings by one user: {user_counts.max():,}")
print(f"Min ratings by one user: {user_counts.min():,}")

plt.figure(figsize=(10, 5))
plt.hist(user_counts.values, bins=50, 
         color='coral', edgecolor='black')
plt.title('User Activity Distribution\n(How many movies each user rated)',
          fontsize=13, fontweight='bold')
plt.xlabel('Number of Ratings per User')
plt.ylabel('Number of Users')
plt.yscale('log')   # Log scale — because most users rate very few
plt.tight_layout()
plt.savefig('processed_data/user_activity.png')
plt.show()
print("✅ Saved user_activity.png")


print("\n📊 Plotting Movie Popularity...")

movie_counts = df['movie_id'].value_counts()

# Top 10 most rated movies
top10 = movie_counts.head(10).reset_index()
top10.columns = ['movie_id', 'count']
top10 = top10.merge(df[['movie_id','title']].drop_duplicates(), 
                    on='movie_id')

print("\nTop 10 Most Rated Movies:")
print(top10[['title', 'count']].to_string(index=False))

plt.figure(figsize=(12, 5))
plt.barh(top10['title'], top10['count'], color='steelblue')
plt.title('Top 10 Most Rated Movies', fontsize=13, fontweight='bold')
plt.xlabel('Number of Ratings')
plt.gca().invert_yaxis()   # Most popular at top
plt.tight_layout()
plt.savefig('processed_data/top_movies.png')
plt.show()
print("✅ Saved top_movies.png")


print("\n📊 Calculating Data Sparsity...")

n_users  = df['user_id'].nunique()
n_movies = df['movie_id'].nunique()
n_ratings = len(df)

total_possible = n_users * n_movies
sparsity = 1 - (n_ratings / total_possible)

print(f"Users:            {n_users:,}")
print(f"Movies:           {n_movies:,}")
print(f"Actual Ratings:   {n_ratings:,}")
print(f"Possible Ratings: {total_possible:,}")
print(f"Sparsity:         {sparsity:.4%}")

# Visual representation
filled = n_ratings / total_possible * 100
empty  = 100 - filled

plt.figure(figsize=(6, 4))
plt.bar(['Rated', 'Not Rated'], [filled, empty], 
        color=['#2ecc71', '#e74c3c'])
plt.title(f'Data Sparsity\n({sparsity:.2%} of ratings are missing)',
          fontsize=13, fontweight='bold')
plt.ylabel('Percentage (%)')
plt.tight_layout()
plt.savefig('processed_data/sparsity.png')
plt.show()
print("✅ Saved sparsity.png")



print("\n📊 Plotting Ratings Over Time...")

yearly = df.groupby('rating_year').size()

plt.figure(figsize=(10, 5))
plt.plot(yearly.index, yearly.values, 
         marker='o', color='purple', linewidth=2)
plt.fill_between(yearly.index, yearly.values, alpha=0.3, color='purple')
plt.title('Number of Ratings Per Year', 
          fontsize=13, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Number of Ratings')
plt.xticks(yearly.index, rotation=45)
plt.tight_layout()
plt.savefig('processed_data/ratings_over_time.png')
plt.show()
print("✅ Saved ratings_over_time.png")



print("\n📊 Average Rating Distribution per Movie...")

movie_avg = df.groupby('movie_id')['rating'].mean()

plt.figure(figsize=(8, 5))
plt.hist(movie_avg.values, bins=30, 
         color='green', edgecolor='black', alpha=0.8)
plt.axvline(movie_avg.mean(), color='red', 
            linestyle='--', label=f'Mean: {movie_avg.mean():.2f}')
plt.title('Average Rating per Movie', 
          fontsize=13, fontweight='bold')
plt.xlabel('Average Rating')
plt.ylabel('Number of Movies')
plt.legend()
plt.tight_layout()
plt.savefig('processed_data/avg_rating_per_movie.png')
plt.show()
print("✅ Saved avg_rating_per_movie.png")