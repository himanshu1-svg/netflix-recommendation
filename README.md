# Netflix Movie Recommendation System

A personalized movie recommendation system built using the 
Netflix Prize Dataset with SVD and Item-Based Collaborative 
Filtering approaches.

## Dataset
Netflix Prize Dataset from Kaggle
- 100 million ratings
- 480,189 users  
- 17,770 movies
- Ratings on 1-5 star scale

Download from: 
https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

Place files in raw_data/ folder before running.

## Project Structure
- change.py      → Load and parse raw Netflix data
- eda.py         → Exploratory Data Analysis with plots
- models.py      → Train SVD and Item-Based CF models
- evaluate.py    → Generate recommendations + evaluate

## Setup Instructions

### 1. Install dependencies
pip install -r requirements.txt

### 2. Download dataset
Download from Kaggle link above.
Place all files inside raw_data/ folder.

### 3. Run in order
python change.py     # Parse data → saves netflix_parsed.csv
python eda.py        # Generate EDA plots
python models.py     # Train models → saves .pkl files
python evaluate.py   # Recommendations + Evaluation

## Results

| Model         | RMSE   | MAP@10 |
|---------------|--------|--------|
| SVD           | 0.9423 | 0.0034 |
| Item-Based CF | 1.0231 | 0.0009 |

SVD outperforms Item-Based CF on both metrics.

## Sample Recommendations

User liked: A Beautiful Mind, Saving Private Ryan
Recommended: Shawshank Redemption, Lord of the Rings, Lost ✅

## Key Findings
- Dataset is 99%+ sparse — main challenge
- SVD captures hidden user taste patterns effectively
- Positivity bias — users mostly rate things they liked
- Power law — few users rate many movies, most rate very few