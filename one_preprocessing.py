import pandas as pd
import numpy as np
import os

movies_df = pd.read_csv("dataset/TMDB_movie_dataset_v11.csv")
credits_df = pd.read_csv("dataset/credits.csv")
combined_df = pd.merge(movies_df, credits_df, left_on='id', right_on='id', how='inner')

print(f"Step 1: movies_df shape: {movies_df.shape}")
print(f"Step 1: credits_df shape: {credits_df.shape}")
print(f"Step 1: combined_df shape: {combined_df.shape}")

# Removing unnecessary columns
keep_columns = [
    'id',
    'title',
    'release_date',
    'revenue',
    'runtime',
    'budget',
    'popularity',
    'spoken_languages',
    'genres',
    'vote_average',
    'cast',
    'crew',
]

combined_df = combined_df[keep_columns]

# Cleaning - removing rows with NaN and revenue == 0
print(f"Step 1: number of rows with at least one NaN: {combined_df.isnull().any(axis=1).sum()}")
print(f"Step 1: number of rows with revenue = 0: {(combined_df['revenue'] == 0).sum()}")

combined_df = combined_df.dropna()
combined_df = combined_df[combined_df['revenue'] != 0]
combined_df = combined_df.reset_index(drop=True)

print(f"Step 1: shape after cleaning: {combined_df.shape}")

# Saving
os.makedirs("df", exist_ok=True)
combined_df.to_csv("df/combined_df.csv", index=False)
print(f"Step 1: saving combined_df to df/combined_df.csv")
print(f"Step 1: completed")
