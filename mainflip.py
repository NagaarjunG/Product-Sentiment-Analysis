import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Print the current working directory
print("Current working directory:", os.getcwd())

# Step 1: Load CSV files
csv_files = [
    'Asus Zenfone_reviews.csv',
    'Micromax Canvas Pro_reviews.csv',
    'Micromax Dual 4_reviews.csv',
    'Motorola Edge 50_reviews.csv',
    'Oneplus Nord 3- 5G_reviews.csv',
    'Samsung Galaxy z flip 5G_reviews.csv'
]

def extract_product_name(dataframe):
    """Extracts the product name from the first row and first column of the dataframe."""
    return dataframe.iloc[0, 0]

# Load dataframes from CSV files
product_reviews = {}
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        product_name = extract_product_name(df)
        product_reviews[product_name] = df
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")

# Sentiment Analysis Function
def analyze_sentiment(review_text):
    """Performs sentiment analysis on the given review text."""
    review_text = str(review_text)
    analysis = TextBlob(review_text)
    polarity_score = analysis.sentiment.polarity
    subjectivity_score = analysis.sentiment.subjectivity

    if polarity_score > 0:
        sentiment = "Positive"
    elif polarity_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return polarity_score, subjectivity_score, sentiment

# Aggregate sentiments from reviews
def aggregate_review_sentiments(reviews):
    """Aggregates sentiment analysis results from a list of reviews."""
    polarities = []
    subjectivities = []
    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

    for review in reviews:
        polarity, subjectivity, sentiment = analyze_sentiment(review)
        polarities.append(polarity)
        subjectivities.append(subjectivity)
        sentiment_counts[sentiment] += 1

    avg_polarity = sum(polarities) / len(polarities) if polarities else 0
    avg_subjectivity = sum(subjectivities) / len(subjectivities) if subjectivities else 0
    dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)

    return avg_polarity, avg_subjectivity, dominant_sentiment, sentiment_counts

# Interpret subjectivity score
def interpret_subjectivity_score(subjectivity):
    """Interprets the subjectivity score into a descriptive label."""
    if subjectivity < 0.2:
        return "Very Objective"
    elif 0.2 <= subjectivity < 0.4:
        return "Objective"
    elif 0.4 <= subjectivity < 0.6:
        return "Neutral"
    elif 0.6 <= subjectivity < 0.8:
        return "Subjective"
    else:
        return "Very Subjective"

# Compare products based on query
def compare_product_sentiments(query):
    """Compares sentiment analysis results for different products based on the given query."""
    sentiments_summary = {}
    doc = nlp(query)
    query_terms = [token.text for token in doc if token.is_alpha]

    vectorizer = TfidfVectorizer().fit([review for df in product_reviews.values() for review in df['Review'].tolist()])

    for product, df in product_reviews.items():
        reviews_list = df['Review'].tolist()

        reviews_tfidf = vectorizer.transform(reviews_list)
        query_tfidf = vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_tfidf, reviews_tfidf).flatten()

        sim_threshold = 0.1  # Adjust threshold as needed
        relevant_reviews = [reviews_list[i] for i in range(len(reviews_list)) if cosine_sim[i] > sim_threshold]

        if relevant_reviews:
            try:
                avg_polarity, avg_subjectivity, dominant_sentiment, sentiment_counts = aggregate_review_sentiments(relevant_reviews)
                sentiments_summary[product] = {
                    'average_polarity': avg_polarity,
                    'average_subjectivity': avg_subjectivity,
                    'overall_sentiment': dominant_sentiment,
                    'Positive_Count': sentiment_counts["Positive"],
                    'Negative_Count': sentiment_counts["Negative"],
                    'Neutral_Count': sentiment_counts["Neutral"]
                }
            except Exception as e:
                st.error(f"Error processing {product}: {e}")

    return sentiments_summary

# Step 5: Streamlit App
def main():
    """Main function to run the Streamlit app."""
    st.image("Text.png", use_column_width=True)
    st.title('Product Sentiment Analysis')

    user_query = st.text_input('Enter your query:')
    if user_query:
        query_sentiment = analyze_sentiment(user_query)
        query_polarity = query_sentiment[0]  # Get polarity score of the query
        print("Query Polarity:", query_polarity)

        sentiment_results = compare_product_sentiments(user_query)
        max_score = max(result["average_polarity"] for result in sentiment_results.values())

        if sentiment_results:
            st.write("Sentiment Analysis Results:")
            best_product = None
            if query_polarity >= 0:
                max_positive_score = -float('inf')
                for product, sentiment in sentiment_results.items():
                    score = sentiment["average_polarity"]
                    subjectivity = sentiment["average_subjectivity"]
                    final_sentiment = sentiment["overall_sentiment"]
                    positive_count = sentiment["Positive_Count"]
                    negative_count = sentiment["Negative_Count"]
                    neutral_count = sentiment["Neutral_Count"]

                    subjectivity_label = interpret_subjectivity_score(subjectivity)

                    if score > max_positive_score:
                        max_positive_score = score
                        best_product = {
                            "Product": product,
                            "Sentiment_Analysis": final_sentiment,
                            "Subjectivity": subjectivity_label,
                            "Score": score,
                            "Positive_Count": positive_count,
                            "Negative_Count": negative_count,
                            "Neutral_Count": neutral_count
                        }

            elif query_polarity < 0:
                max_negative_score = float('inf')
                for product, sentiment in sentiment_results.items():
                    score = sentiment["average_polarity"]
                    subjectivity = sentiment["average_subjectivity"]
                    final_sentiment = sentiment["overall_sentiment"]
                    positive_count = sentiment["Positive_Count"]
                    negative_count = sentiment["Negative_Count"]
                    neutral_count = sentiment["Neutral_Count"]

                    subjectivity_label = interpret_subjectivity_score(subjectivity)

                    if score < max_negative_score:
                        max_negative_score = score
                        best_product = {
                            "Product": product,
                            "Sentiment_Analysis": final_sentiment,
                            "Subjectivity": subjectivity_label,
                            "Score": score,
                            "Positive_Count": positive_count,
                            "Negative_Count": negative_count,
                            "Neutral_Count": neutral_count
                        }

            if best_product:
                sentiment_summary = pd.DataFrame([best_product])
                st.write(sentiment_summary)

            # Visualization (optional): Bar chart of average polarities
            product_names = list(sentiment_results.keys())
            scores = [sentiment['average_polarity'] for sentiment in sentiment_results.values()]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(product_names, scores, color='salmon')
            ax.set_xlabel('Products')
            ax.set_ylabel('Average Polarity')
            ax.set_title('Average Polarity of Products')

            # Set x-axis ticks and labels
            ax.set_xticks(range(len(product_names)))
            ax.set_xticklabels(product_names, rotation=45, ha='right')

            st.pyplot(fig)

        else:
            st.write("No sentiment analysis results found.")

if __name__ == "__main__":
    main()
