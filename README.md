# Content-Based Recommendation System

Hi, I'm **Shambhavi M Puttane**, a Master's student in Data Science. This project is a simple yet effective content-based recommendation system built as part of an AI/Machine Learning Intern Challenge. It uses **TF-IDF** and **cosine similarity** to recommend movies based on a user's text input.

### Salary Expectation - 2500- 3000$/month

## Demo Video
[![Project Demo]([https://img.shields.io/badge/Watch-Demo-red](https://drive.google.com/file/d/1Bvj9kkZNgNSrtsEmh91s9y854RrDQ-9P/view?usp=sharing))]([video_link_here](https://drive.google.com/file/d/1Bvj9kkZNgNSrtsEmh91s9y854RrDQ-9P/view?usp=sharing))

Check out the video demonstration of the recommendation system in action [here]([video_link_here](https://drive.google.com/file/d/1Bvj9kkZNgNSrtsEmh91s9y854RrDQ-9P/view?usp=sharing)).

## Dataset
- **Source**: [TMDB Movies Metadata](https://www.kaggle.com/tmdb/tmdb-movie-metadata) from Kaggle.
- **Steps**:
 1. Download `movies_metadata.csv`
 2. Place it in the `data` folder

## Setup
1. **Python**: 3.6+
2. **Dependencies**: Install with `pip install -r requirements.txt`

## Usage
Run the script with a query:
```bash
python recommend.py "I like action movies with comedy" 
```
Processing query: 'I like action movies with comedy'
Dataset loaded and preprocessed successfully!
Recommendations generated successfully!

Top 5 Recommendations:
Mr. & Mrs. Smith (Score: 0.1076)
The 6th Day (Score: 0.0827)
Top Cat Begins (Score: 0.0790)
Mad Max: Fury Road (Score: 0.0789)
Zookeeper (Score: 0.0775)
