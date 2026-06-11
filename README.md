# 🎬 Netflix Movie Recommendation System

> A personalized movie recommendation engine built using the Netflix Prize Dataset.  
> Compares **SVD** and **Item-Based Collaborative Filtering** with full evaluation using RMSE and MAP@10.

---

## 👥 Team Members

| Name | Roll Number |
|------|-------------|
| Himanshu | 25321014 |
| Chanchal Choudhary | 25114023 |

---

## 📌 Table of Contents

- [What This Project Does](#what-this-project-does)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Results](#results)
- [Sample Recommendations](#sample-recommendations)
- [Key Insights](#key-insights)
- [Future Work](#future-work)

---

## 🎯 What This Project Does

This project builds a **personalized movie recommendation system** that:

- Reads 100 million real user ratings from the Netflix Prize Dataset
- Learns hidden patterns in user preferences using **Machine Learning**
- Predicts how a user would rate a movie they haven't seen yet
- Generates a **Top-10 personalized movie list** for any user
- Compares two different recommendation approaches and picks the best one

Think of it like building your own mini Netflix recommendation engine from scratch.

---

## 📦 Dataset

**Source:** [Netflix Prize Dataset on Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

| Property | Value |
|----------|-------|
| Total Ratings | 100,480,507 |
| Unique Users | 480,189 |
| Unique Movies | 17,770 |
| Rating Scale | 1 to 5 Stars |
| Time Period | 1999 – 2005 |

### Files in the Dataset

| File | Description |
|------|-------------|
| `combined_data_1.txt` | Movie ratings — Part 1 (~24M ratings) |
| `combined_data_2.txt` | Movie ratings — Part 2 (~25M ratings) |
| `combined_data_3.txt` | Movie ratings — Part 3 (~26M ratings) |
| `combined_data_4.txt` | Movie ratings — Part 4 (~25M ratings) |
| `movie_titles.csv` | Movie ID, Release Year, Title |

### Raw Data Format

The rating files have an unusual format — movie ID is on its own line:

```
1:
1488844,3,2005-09-06
822109,5,2005-05-13
885013,4,2005-10-19
2:
301154,1,2005-10-19
...
```

This means: User `1488844` gave Movie `1` a rating of `3` on `2005-09-06`.  
Our parser in `change.py` handles this format automatically.

> ⚠️ Due to computational constraints, we used a **sample of 2,000,000 ratings** which is representative of the full dataset.

---

## 🗂️ Project Structure

```
netflix-recommendation/
│
├── 📄 change.py              ← STEP 1: Parse raw data files → save CSV
├── 📄 eda.py                 ← STEP 2: Exploratory Data Analysis + plots
├── 📄 models.py              ← STEP 3: Train SVD and Item-CF models
├── 📄 evaluate.py            ← STEP 4: Generate recommendations + evaluate
│
├── 📁 raw_data/              ← Put downloaded Kaggle files here
│   ├── combined_data_1.txt
│   ├── combined_data_2.txt
│   ├── combined_data_3.txt
│   ├── combined_data_4.txt
│   └── movie_titles.csv
│
├── 📁 processed_data/        ← Auto-generated after running change.py
│   ├── netflix_parsed.csv        ← cleaned 2M row dataset
│   ├── model_results.csv         ← RMSE and MAP@10 scores
│   ├── rating_distribution.png   ← EDA plot
│   ├── user_activity.png         ← EDA plot
│   ├── top_movies.png            ← EDA plot
│   ├── sparsity.png              ← EDA plot
│   ├── ratings_over_time.png     ← EDA plot
│   └── avg_rating_per_movie.png  ← EDA plot
│
├── 📁 models/                ← Auto-generated after running models.py
│   ├── svd_model.pkl             ← trained SVD model
│   └── item_cf_model.pkl         ← trained Item-CF model
│
├── 📄 requirements.txt       ← Python libraries needed
└── 📄 README.md              ← This file
```

> 📝 `raw_data/` and `models/` are NOT uploaded to GitHub (too large).  
> They are generated locally by running the scripts in order.

---

## ⚙️ Setup Instructions

### Step 1 — Clone the Repository

```bash
git clone https://github.com/himanshu1-svg/netflix-recommendation.git
cd netflix-recommendation
```

### Step 2 — Install Python (if not already installed)

Download from: https://www.python.org/downloads/  
Make sure you have **Python 3.8 or higher**.

Check your version:
```bash
python --version
```

### Step 3 — Install Required Libraries

```bash
pip install -r requirements.txt
```

This installs:

| Library | What it does |
|---------|--------------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical calculations |
| `matplotlib` | Drawing charts and plots |
| `seaborn` | Prettier chart styling |
| `scikit-surprise` | Recommendation system algorithms (SVD, CF) |
| `scikit-learn` | Machine learning utilities |

### Step 4 — Download the Dataset

1. Go to: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
2. Click **Download**
3. Extract the zip file
4. Place all files inside the `raw_data/` folder:

```
raw_data/
├── combined_data_1.txt
├── combined_data_2.txt
├── combined_data_3.txt
├── combined_data_4.txt
└── movie_titles.csv
```

---

## ▶️ How to Run

Run the scripts **in this exact order**. Each step depends on the previous one.

---

### 🔵 STEP 1 — Parse & Load Data

```bash
python change.py
```

**What it does:**
- Opens all 4 raw Netflix data files
- Reads 100 million rating rows line by line
- Parses the unusual format (movie ID on its own line)
- Samples 2 million rows for efficiency
- Loads movie titles from `movie_titles.csv`
- Merges ratings with movie titles
- Saves the result to `processed_data/netflix_parsed.csv`

**Expected output:**
```
Loading file 1...
Loading file 2...
Loading file 3...
Loading file 4...
Total ratings: (100480507, 4)
Sampled: (2000000, 4)
Loading movie titles...
✅ Movies loaded: (17770, 3)
Merged shape: (2000000, 6)
Missing values: all 0
Rating range: 1 to 5
✅ Data saved to processed_data/netflix_parsed.csv
```

**Time:** ~10–15 minutes (reads 100M rows)

---

### 🟡 STEP 2 — Exploratory Data Analysis

```bash
python eda.py
```

**What it does:**
- Loads the saved CSV (fast — seconds not minutes)
- Calculates basic statistics (total ratings, users, movies)
- Generates 6 plots and saves them as PNG files
- Prints key insights about the data

**Plots generated:**

| Plot | Insight |
|------|---------|
| `rating_distribution.png` | Most users give 4-5 stars (positivity bias) |
| `user_activity.png` | Few users rate a lot, most rate very little (power law) |
| `top_movies.png` | Lord of the Rings, Shawshank dominate |
| `sparsity.png` | 99.96% of ratings are missing |
| `ratings_over_time.png` | Rating activity grew 2003–2005 |
| `avg_rating_per_movie.png` | Most movies average around 3.5–4 stars |

**Time:** ~1–2 minutes

---

### 🟠 STEP 3 — Train Models

```bash
python models.py
```

**What it does:**
- Loads the saved CSV
- Splits data: 80% training, 20% testing
- Trains **Model 1: SVD** (Singular Value Decomposition)
- Trains **Model 2: Item-Based Collaborative Filtering**
- Evaluates both on RMSE and MAE
- Prints a comparison table
- Saves both models as `.pkl` files

**Models explained:**

**SVD (Singular Value Decomposition)**
- Breaks down the user-movie rating matrix into hidden "taste factors"
- Each user gets 50 hidden preference dimensions
- Each movie gets 50 hidden feature dimensions
- Predicted rating = how well user taste matches movie features
- Very fast and handles sparse data well

**Item-Based Collaborative Filtering**
- Finds movies that are similar to what the user has already liked
- Uses cosine similarity between movie rating vectors
- Considers the 40 most similar movies
- More explainable but slower to train

**Expected output:**
```
Training MODEL 1: SVD
✅ SVD Training done!
RMSE: 0.9423
MAE:  0.7401

Training MODEL 2: Item-Based CF
✅ Item-CF Training done!
RMSE: 1.0231
MAE:  0.8102

MODEL COMPARISON:
        Model    RMSE     MAE
          SVD  0.9423  0.7401
Item-Based CF  1.0231  0.8102

🏆 Better Model (lower RMSE): SVD
✅ Models saved to models/ folder
```

**Time:** SVD ~5 min, Item-CF ~15–20 min

---

### 🟢 STEP 4 — Generate Recommendations & Evaluate

```bash
python evaluate.py
```

**What it does:**
- Loads both saved models
- Picks 3 random users from the dataset
- For each user → predicts ratings for all unrated movies → returns Top 10
- Shows what the user liked and what was recommended
- Computes **MAP@10** for both models on 200 users
- Saves final comparison table to CSV

**What is MAP@10?**  
MAP@10 (Mean Average Precision at 10) measures whether the top 10 recommendations actually contain movies the user would enjoy.  
A movie counts as "relevant" if the user actually rated it **3.5 stars or above**.

**Expected output:**
```
👤 User 2313030
   Liked: ['A Beautiful Mind', 'Saving Private Ryan', '50 First Dates']
   Top 10 Recommendations:
    1. The Shawshank Redemption    (predicted: ⭐4.86)
    2. Lord of the Rings Extended  (predicted: ⭐4.88)
    ...

✅ SVD     MAP@10: 0.0034
✅ Item-CF MAP@10: 0.0009

🏆 Better RMSE:   SVD
🏆 Better MAP@10: SVD
✅ Results saved to processed_data/model_results.csv
```

**Time:** ~10–25 minutes (predicts ratings for many users)

---

## 📊 Results

| Model | RMSE | MAE | MAP@10 |
|-------|------|-----|--------|
| **SVD** | **0.9423** ✓ | **0.7401** ✓ | **0.0034** ✓ |
| Item-Based CF | 1.0231 | 0.8102 | 0.0009 |

**SVD wins on all three metrics.**

### Why is MAP@10 low?
This is expected and consistent with research on the Netflix Prize dataset.  
With 99.96% data sparsity and very few ratings per user, finding the exact relevant movies in the top 10 is inherently difficult.  
Even so, SVD achieves **3.7x better MAP@10** than Item-CF.

---

## 🎬 Sample Recommendations

### User 2313030
**Liked:** A Beautiful Mind, Saving Private Ryan, 50 First Dates

| Rank | Recommended Movie | Predicted Rating |
|------|-------------------|-----------------|
| 1 | The West Wing: Season 3 | ⭐ 4.90 |
| 2 | Lord of the Rings: Two Towers Extended | ⭐ 4.88 |
| 3 | The Shawshank Redemption | ⭐ 4.86 |
| 4 | Arrested Development: Season 2 | ⭐ 4.88 |
| 5 | Lost: Season 1 | ⭐ 4.75 |

✅ Drama/war film lover → got critically acclaimed dramas

---

### User 1974299
**Liked:** Eternal Sunshine of the Spotless Mind, The Fugitive, American Splendor

| Rank | Recommended Movie | Predicted Rating |
|------|-------------------|-----------------|
| 1 | Lord of the Rings: Fellowship | ⭐ 4.90 |
| 2 | Forrest Gump | ⭐ 4.85 |
| 3 | The Godfather | ⭐ 4.80 |
| 4 | Star Wars: Episode V | ⭐ 4.78 |
| 5 | 24: Season 2 | ⭐ 4.84 |

✅ Thoughtful film lover → got timeless classics

---

## 💡 Key Insights

| Insight | What It Means |
|---------|---------------|
| **Positivity Bias** | Users mostly rate movies they enjoyed — dataset skews toward 4-5 stars |
| **Power Law Activity** | A few users rate thousands of movies; most rate very few |
| **99.96% Sparsity** | Almost all user-movie combinations are empty — the core challenge |
| **Popularity Bias** | Popular movies (LOTR, Shawshank) dominate recommendations |
| **SVD superiority** | Matrix factorization handles sparsity much better than CF |

---

## 🔮 Future Work

- [ ] Train on full 100 million ratings for better accuracy
- [ ] Implement Neural Collaborative Filtering (deep learning)
- [ ] Build a Hybrid system combining SVD + content features (genre, director)
- [ ] Handle Cold Start problem for new users
- [ ] Add diversity metrics to reduce popularity bias
- [ ] Deploy as a live web application or REST API
- [ ] Build an interactive dashboard for exploring recommendations

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Programming language |
| Pandas | Data loading and manipulation |
| Matplotlib + Seaborn | Data visualization |
| Scikit-Surprise | SVD and Collaborative Filtering |
| Scikit-Learn | Train-test split, utilities |
| Pickle | Saving trained models |

---

## 📁 Reproducing Results

To fully reproduce our results from scratch:

```bash
# 1. Clone repo
git clone https://github.com/himanshu1-svg/netflix-recommendation.git
cd netflix-recommendation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place in raw_data/

# 4. Run all steps in order
python change.py      # ~10-15 min
python eda.py         # ~1-2 min
python models.py      # ~20-25 min
python evaluate.py    # ~15-25 min
```

All outputs (CSV files, plots, model files) will be generated automatically.

---

## 📬 Contact

| Member | Roll Number |
|--------|-------------|
| Himanshu | 25321014 |
| Chanchal Choudhary | 25114023 |

**Repository:** https://github.com/himanshu1-svg/netflix-recommendation

---

*Built for AI/ML Hackathon — June 2026*