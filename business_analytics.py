

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
print("Loading data...")
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
tags = pd.read_csv("tags.csv")

# Convert timestamp to datetime
ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['year'] = ratings['datetime'].dt.year
ratings['month'] = ratings['datetime'].dt.month
tags['datetime'] = pd.to_datetime(tags['timestamp'], unit='s')

print(f"Data loaded: {len(ratings)} ratings, {len(movies)} movies, {len(tags)} tags")
# 1.1 Harsh vs Generous Raters
print("\n1.1 Identifying Harsh vs Generous Raters...")

user_stats = ratings.groupby('userId').agg({
    'rating': ['mean', 'count', 'std']
}).reset_index()
user_stats.columns = ['userId', 'mean_rating', 'rating_count', 'rating_std']

# Filter users with at least 20 ratings for reliable statistics
active_users = user_stats[user_stats['rating_count'] >= 20].copy()

# Define categories
global_mean = ratings['rating'].mean()
active_users['rater_type'] = pd.cut(
    active_users['mean_rating'], 
    bins=[0, global_mean - 0.5, global_mean + 0.5, 5],
    labels=['Harsh', 'Average', 'Generous']
)

rater_distribution = active_users['rater_type'].value_counts()
print("\nRater Distribution (users with 20+ ratings):")
print(rater_distribution)
print(f"\nGlobal Mean Rating: {global_mean:.3f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
axes[0].pie(rater_distribution, labels=rater_distribution.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Distribution of Rater Types')

# Box plot
sns.boxplot(data=active_users, x='rater_type', y='mean_rating', ax=axes[1])
axes[1].set_title('Mean Rating Distribution by Rater Type')
axes[1].set_ylabel('Mean Rating')
axes[1].axhline(global_mean, color='red', linestyle='--', label=f'Global Mean ({global_mean:.2f})')
axes[1].legend()

plt.tight_layout()
plt.savefig('insights_rater_types.png', dpi=300, bbox_inches='tight')
print(" Saved: insights_rater_types.png")
plt.close()

# 1.2 Rating Evolution Over Time
print("\n1.2 Analyzing Rating Trends Over Time...")

yearly_ratings = ratings.groupby('year').agg({
    'rating': ['mean', 'count']
}).reset_index()
yearly_ratings.columns = ['year', 'mean_rating', 'count']

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Average rating over time
axes[0].plot(yearly_ratings['year'], yearly_ratings['mean_rating'], marker='o', linewidth=2)
axes[0].set_title('Average Movie Rating Over Time', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Average Rating')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(global_mean, color='red', linestyle='--', alpha=0.5, label=f'Overall Mean ({global_mean:.2f})')
axes[0].legend()

# Number of ratings over time
axes[1].bar(yearly_ratings['year'], yearly_ratings['count'], alpha=0.7)
axes[1].set_title('Number of Ratings Over Time', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Ratings')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('insights_rating_evolution.png', dpi=300, bbox_inches='tight')
print(" Saved: insights_rating_evolution.png")
plt.close()

# 1.3 User Activity Retention
print("\n1.3 Analyzing User Activity and Retention...")

# Calculate first and last rating date for each user
user_activity = ratings.groupby('userId').agg({
    'datetime': ['min', 'max'],
    'rating': 'count'
}).reset_index()
user_activity.columns = ['userId', 'first_rating', 'last_rating', 'total_ratings']

# Calculate days active
user_activity['days_active'] = (user_activity['last_rating'] - user_activity['first_rating']).dt.days

# Activity categories
user_activity['activity_level'] = pd.cut(
    user_activity['total_ratings'],
    bins=[0, 20, 50, 100, 200, float('inf')],
    labels=['Casual (1-20)', 'Regular (21-50)', 'Active (51-100)', 'Very Active (101-200)', 'Power User (200+)']
)

activity_dist = user_activity['activity_level'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
activity_dist.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('User Activity Level Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Activity Level')
plt.ylabel('Number of Users')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('insights_user_activity.png', dpi=300, bbox_inches='tight')
print(" Saved: insights_user_activity.png")
plt.close()


# 2.1 Genre Performance

# Parse genres
movies_exploded = movies.copy()
movies_exploded['genres'] = movies_exploded['genres'].fillna('Unknown')
movies_exploded = movies_exploded[movies_exploded['genres'] != '(no genres listed)']

# Expand genres
genre_list = []
for idx, row in movies_exploded.iterrows():
    genres = row['genres'].split('|')
    for genre in genres:
        genre_list.append({'movieId': row['movieId'], 'genre': genre})

genre_df = pd.DataFrame(genre_list)

# Merge with ratings
genre_ratings = genre_df.merge(ratings[['movieId', 'rating']], on='movieId')

# Calculate stats
genre_stats = genre_ratings.groupby('genre').agg({
    'rating': ['mean', 'count', 'std']
}).reset_index()
genre_stats.columns = ['genre', 'mean_rating', 'count', 'std']

# Filter genres with at least 1000 ratings
popular_genres = genre_stats[genre_stats['count'] >= 1000].sort_values('mean_rating', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Best performing genres
top_genres = popular_genres.head(10)
axes[0].barh(top_genres['genre'], top_genres['mean_rating'], color='green', alpha=0.7)
axes[0].set_xlabel('Average Rating')
axes[0].set_title('Top 10 Best Performing Genres', fontsize=12, fontweight='bold')
axes[0].axvline(global_mean, color='red', linestyle='--', alpha=0.5, label=f'Global Mean ({global_mean:.2f})')
axes[0].legend()
axes[0].invert_yaxis()

# Worst performing genres
bottom_genres = popular_genres.tail(10).sort_values('mean_rating')
axes[1].barh(bottom_genres['genre'], bottom_genres['mean_rating'], color='red', alpha=0.7)
axes[1].set_xlabel('Average Rating')
axes[1].set_title('Bottom 10 Worst Performing Genres', fontsize=12, fontweight='bold')
axes[1].axvline(global_mean, color='red', linestyle='--', alpha=0.5, label=f'Global Mean ({global_mean:.2f})')
axes[1].legend()
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('insights_genre_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: insights_genre_performance.png")
plt.close()

# 2.2 Tag-driven Analytics
print("\n2.2 Analyzing Tags...")

# Most common tags
tag_counts = tags['tag'].str.lower().value_counts().head(20)

plt.figure(figsize=(12, 6))
tag_counts.plot(kind='barh', color='purple', alpha=0.7)
plt.title('Top 20 Most Common Movie Tags', fontsize=14, fontweight='bold')
plt.xlabel('Frequency')
plt.ylabel('Tag')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('insights_top_tags.png', dpi=300, bbox_inches='tight')
print("✓ Saved: insights_top_tags.png")
plt.close()

# Tag sentiment analysis (simple keyword-based)
positive_keywords = ['great', 'excellent', 'amazing', 'love', 'best', 'favorite', 'masterpiece', 'brilliant']
negative_keywords = ['bad', 'boring', 'worst', 'terrible', 'awful', 'disappointing', 'waste']
neutral_keywords = ['classic', 'action', 'comedy', 'drama', 'thriller', 'suspense']

tags['tag_lower'] = tags['tag'].str.lower()
tags['sentiment'] = 'Neutral'
tags.loc[tags['tag_lower'].str.contains('|'.join(positive_keywords), na=False), 'sentiment'] = 'Positive'
tags.loc[tags['tag_lower'].str.contains('|'.join(negative_keywords), na=False), 'sentiment'] = 'Negative'

sentiment_dist = tags['sentiment'].value_counts()

plt.figure(figsize=(8, 6))
plt.pie(sentiment_dist, labels=sentiment_dist.index, autopct='%1.1f%%', startangle=90, 
        colors=['#2ecc71', '#e74c3c', '#95a5a6'])
plt.title('Tag Sentiment Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('insights_tag_sentiment.png', dpi=300, bbox_inches='tight')
print("✓ Saved: insights_tag_sentiment.png")
plt.close()

# 3.1 Release Year Impact
print("\n3.1 Analyzing Release Year Impact on Ratings...")

# Extract year from movie title
movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
movies_with_year = movies.dropna(subset=['release_year'])

# Merge with ratings
movie_ratings = ratings.groupby('movieId').agg({
    'rating': ['mean', 'count']
}).reset_index()
movie_ratings.columns = ['movieId', 'mean_rating', 'rating_count']

movies_year_ratings = movies_with_year.merge(movie_ratings, on='movieId')
movies_year_ratings = movies_year_ratings[movies_year_ratings['rating_count'] >= 20]  # Filter well-rated movies

# Group by decade
movies_year_ratings['decade'] = (movies_year_ratings['release_year'] // 10 * 10).astype(int)
decade_stats = movies_year_ratings.groupby('decade').agg({
    'mean_rating': 'mean',
    'rating_count': 'sum'
}).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Average rating by decade
axes[0].plot(decade_stats['decade'], decade_stats['mean_rating'], marker='o', linewidth=2, markersize=8)
axes[0].set_title('Average Movie Rating by Decade', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Decade')
axes[0].set_ylabel('Average Rating')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(global_mean, color='red', linestyle='--', alpha=0.5, label=f'Global Mean ({global_mean:.2f})')
axes[0].legend()

# Number of ratings by decade
axes[1].bar(decade_stats['decade'], decade_stats['rating_count'], alpha=0.7, color='coral')
axes[1].set_title('Total Ratings by Movie Decade', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Decade')
axes[1].set_ylabel('Total Ratings')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('insights_release_year_impact.png', dpi=300, bbox_inches='tight')
print(" Saved: insights_release_year_impact.png")
plt.close()

# 3.2 Hidden Gems (High ratings, low visibility)
print("\n3.2 Discovering Hidden Gems...")

# Define hidden gems: high rating (>4.0) but low number of ratings (<100)
hidden_gems = movies_year_ratings[
    (movies_year_ratings['mean_rating'] >= 4.0) & 
    (movies_year_ratings['rating_count'] < 100) &
    (movies_year_ratings['rating_count'] >= 20)  # Ensure some reliability
].sort_values('mean_rating', ascending=False).head(20)

print("\nTop 20 Hidden Gems:")
print(hidden_gems[['title', 'genres', 'mean_rating', 'rating_count']])

# Save to CSV
hidden_gems[['title', 'genres', 'release_year', 'mean_rating', 'rating_count']].to_csv(
    'hidden_gems.csv', index=False
)
print(" Saved: hidden_gems.csv")

# Visualize
plt.figure(figsize=(14, 8))
plt.scatter(movies_year_ratings['rating_count'], movies_year_ratings['mean_rating'], 
            alpha=0.3, s=10, label='All Movies')
plt.scatter(hidden_gems['rating_count'], hidden_gems['mean_rating'], 
            color='red', s=100, marker='*', label='Hidden Gems', zorder=5)
plt.xlabel('Number of Ratings', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.title('Hidden Gems: High Rating, Low Visibility', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')  # Log scale for better visualization
plt.tight_layout()
plt.savefig('insights_hidden_gems.png', dpi=300, bbox_inches='tight')
print(" Saved: insights_hidden_gems.png")
plt.close()

# 3.3 Genre Affinity Clusters
print("\n3.3 Analyzing Genre Affinity Clusters...")

# Create user-genre matrix
user_genre_ratings = ratings.merge(genre_df, on='movieId')
user_genre_avg = user_genre_ratings.groupby(['userId', 'genre'])['rating'].mean().reset_index()

# Pivot to create user-genre matrix
user_genre_matrix = user_genre_avg.pivot_table(
    index='userId', 
    columns='genre', 
    values='rating'
).fillna(user_genre_avg['rating'].mean())

# Take sample of users for clustering
sample_size = min(5000, len(user_genre_matrix))
user_sample = user_genre_matrix.sample(n=sample_size, random_state=42)

# Calculate correlation between genres
genre_corr = user_genre_matrix.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(genre_corr, cmap='coolwarm', center=0, square=True, 
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Genre Affinity Heatmap (User Rating Correlations)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('insights_genre_affinity.png', dpi=300, bbox_inches='tight')
print(" Saved: insights_genre_affinity.png")
plt.close()


print("SUMMARY STATISTICS")


summary = {
    'Total Movies': len(movies),
    'Total Ratings': len(ratings),
    'Total Users': ratings['userId'].nunique(),
    'Total Tags': len(tags),
    'Global Mean Rating': round(global_mean, 3),
    'Harsh Raters (%)': round(rater_distribution.get('Harsh', 0) / len(active_users) * 100, 1),
    'Generous Raters (%)': round(rater_distribution.get('Generous', 0) / len(active_users) * 100, 1),
    'Power Users (200+ ratings)': len(user_activity[user_activity['activity_level'] == 'Power User (200+)']),
    'Top Genre': popular_genres.iloc[0]['genre'],
    'Hidden Gems Found': len(hidden_gems)
}

print("\n Key Metrics:")
for key, value in summary.items():
    print(f"  {key}: {value}")

# Save summary to file
with open('insights_summary.txt', 'w') as f:
    f.write("MOVIE RECOMMENDER SYSTEM - BUSINESS INSIGHTS SUMMARY\n")
    f.write("="*60 + "\n\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")
print("\n Saved: insights_summary.txt")

print("ALL BUSINESS INSIGHTS GENERATED SUCCESSFULLY!")

print("\nGenerated files:")
print("  - insights_rater_types.png")
print("  - insights_rating_evolution.png")
print("  - insights_user_activity.png")
print("  - insights_genre_performance.png")
print("  - insights_top_tags.png")
print("  - insights_tag_sentiment.png")
print("  - insights_release_year_impact.png")
print("  - insights_hidden_gems.png")
print("  - insights_genre_affinity.png")
print("  - hidden_gems.csv")
print("  - insights_summary.txt")
