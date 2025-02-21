import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    """Preprocess text using NLTK."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = nltk.word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()  # Lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def load_data(filepath):
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv(filepath, low_memory=False)
        df = df[['title', 'overview']].dropna()  # Keep only relevant columns
        df['overview'] = df['overview'].apply(preprocess_text)  # Preprocess text
        df = df.head(500)  # Limit to 500 rows for simplicity
        print("Dataset loaded and preprocessed successfully!")
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found. Please ensure the file exists in the 'data' folder.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    return df

def recommend(query, df, vectorizer):
    """Compute similarity and return top 5 recommendations."""
    try:
        query_vec = vectorizer.transform([query])  # Transform query into TF-IDF vector
        similarities = cosine_similarity(query_vec, vectorizer.transform(df['overview'])).flatten()
        df['similarity'] = similarities  # Add similarity scores to the dataframe
        df_sorted = df.sort_values(by='similarity', ascending=False).head(5)  # Sort by similarity
        print("Recommendations generated successfully!")
    except Exception as e:
        print(f"An unexpected error occurred while generating recommendations: {e}")
        sys.exit(1)
    return df_sorted[['title', 'similarity']]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recommend.py 'Your query here'")
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    print(f"Processing query: '{query}'")

    # Load and preprocess the dataset
    df = load_data('data/movies_metadata.csv')

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))  # Use unigrams and bigrams
    vectorizer.fit(df['overview'])

    # Generate recommendations
    recommendations = recommend(query, df, vectorizer)

    # Display the results
    print("\nTop 5 Recommendations:")
    for idx, row in recommendations.iterrows():
        print(f"{row['title']} (Score: {row['similarity']:.4f})")