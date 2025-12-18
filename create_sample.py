import pandas as pd

print("Loading full dataset...")
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print(f"Original: {len(ratings):,} ratings")

# Sample 500,000 ratings (plenty for demo!)
print("Creating sample...")
ratings_sample = ratings.sample(n=500000, random_state=42)

# Get only movies that appear in the sample
rated_movie_ids = ratings_sample['movieId'].unique()
movies_sample = movies[movies['movieId'].isin(rated_movie_ids)]

# Save smaller files
print("Saving smaller files...")
ratings_sample.to_csv("ratings_small.csv", index=False)
movies_sample.to_csv("movies_small.csv", index=False)

print(f"\n Created smaller files:")
print(f"   - ratings_small.csv: {len(ratings_sample):,} ratings")
print(f"   - movies_small.csv: {len(movies_sample):,} movies")

# Check file sizes
import os
ratings_size = os.path.getsize("ratings_small.csv") / (1024 * 1024)
movies_size = os.path.getsize("movies_small.csv") / (1024 * 1024)

print(f"\n File sizes:")
print(f"   - ratings_small.csv: {ratings_size:.1f} MB")
print(f"   - movies_small.csv: {movies_size:.1f} MB")

if ratings_size < 100 and movies_size < 100:
    print("\n Both files are under 100MB! Ready for GitHub!")
else:
    print("\n Still too big. Try 250,000 ratings instead.")