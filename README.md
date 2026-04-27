# MATH UN2015 Final Project: Effectiveness of SVD in Movie Recommendations

A collaborative filtering movie recommendation system built from scratch using Singular Value Decomposition (SVD) on the MovieLens dataset.

## Overview

This project implements SVD-based matrix factorization to predict user movie ratings. It compares two approaches — raw SVD and mean-centered SVD — against naive baselines, and explores what latent factors the model discovers.

## How It Works

1. **Build a user-movie matrix** from MovieLens ratings, filtering out users with fewer than 50 ratings.
2. **Train/test split** — 10% of known ratings are held out for evaluation.
3. **Decompose the matrix** using eigendecomposition of AᵀA to compute U, Σ, and Vᵀ.
4. **Mean-centered SVD** — subtract each user's mean rating before decomposition, then add it back at prediction time to reduce personal rating bias.
5. **Evaluate over k = 2–50** latent factors using RMSE, comparing raw SVD, mean-centered SVD, global mean baseline, and user mean baseline.
6. **Inspect latent factors** — the top/bottom movies per factor reveal what taste dimensions the model has learned.

## Requirements

```
pandas
numpy
matplotlib
```

The dataset requires two CSV files in the working directory (these can be found in the ml-latest-small folder in this repository:
- `ratings.csv` — columns: `userId`, `movieId`, `rating`, `timestamp`
- `movies.csv` — columns: `movieId`, `title`, `genres`

## Usage

Open and run `MovieLensSVD.ipynb` top to bottom. The notebook will:

- Print the user-movie matrix shape and sparsity
- Plot RMSE vs. k for both SVD variants
- Print a summary table comparing all four methods at their best k
- Print the top and bottom 15 movies for each of the first 5 latent factors

## Results

Mean-centered SVD consistently outperforms raw SVD by correcting for the fact that different users have different rating scales (e.g., a 4 from a harsh rater means more than a 4 from someone who rates everything highly). Both SVD variants outperform the global mean and user mean baselines at the right value of k.

## Project Structure

```
MovieLensSVD.ipynb   # Main notebook
ml-latest-small/
  ratings.csv          # MovieLens ratings data 
  movies.csv           # MovieLens movie metadata
```
