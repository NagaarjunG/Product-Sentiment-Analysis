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
    'csv-1',
    'csv-2',
    'csv-3',
    'csv-4',
    'csv-5',
    'csv-6'
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


def main():
    """Main function to run the Streamlit app."""
    # Set the page config
    st.set_page_config(
        page_title="Product Sentiment Analysis",
        page_icon="📊",
        layout="wide"
    )

    # Custom CSS for styling (optional)
    st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 2em;
        }
        .title {
            color: #1f77b4;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 0.5em;
        }
        .query-input {
            font-size: 1.5em;
            padding: 0.5em;
            border: 2px solid #1f77b4;
            border-radius: 5px;
            margin-bottom: 1em;
        }
        .result-table {
            background-color: #fff;
            padding: 1em;
            border-radius: 5px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
            margin-top: 1em;
        }
    </style>
    """, unsafe_allow_html=True)

    st.image("Mylogo.png", width=200)
    st.markdown('<div class="title">Product Sentiment Analysis</div>', unsafe_allow_html=True)

    # # Displaying the label with custom styling
    st.markdown('<span style="color: #1f77b4; font-size: 1.2em;">Enter your query:</span>', unsafe_allow_html=True)

    # Input query from the user
    user_query = st.text_input('', placeholder='Type your query here...', key="query")
    if user_query:
        query_sentiment = analyze_sentiment(user_query)
        query_polarity = query_sentiment[0]  # Get polarity score of the query
        # st.write(f"**Query Polarity:** {query_polarity}")

        sentiment_results = compare_product_sentiments(user_query)
        if sentiment_results:
            st.markdown('<h3 style="color: #1f77b4;">Sentiment Analysis Results:</h3>', unsafe_allow_html=True)
            best_product = None
            max_score = -float('inf') if query_polarity >= 0 else float('inf')

            for product, sentiment in sentiment_results.items():
                score = sentiment["average_polarity"]
                subjectivity = sentiment["average_subjectivity"]
                final_sentiment = sentiment["overall_sentiment"]
                positive_count = sentiment["Positive_Count"]
                negative_count = sentiment["Negative_Count"]
                neutral_count = sentiment["Neutral_Count"]

                if (query_polarity >= 0 and score > max_score) or (query_polarity < 0 and score < max_score):
                    max_score = score
                    best_product = {
                        "Product": product,
                        "Sentiment_Analysis": final_sentiment,
                        "Score": score,
                        "Positive_Count": positive_count,
                        "Negative_Count": negative_count,
                        "Neutral_Count": neutral_count
                    }

            if best_product:
                sentiment_summary = pd.DataFrame([best_product])
                st.write(sentiment_summary.style.set_table_styles(
                    [{'selector': 'th', 'props': [('background-color', '#1f77b4'), ('color', 'white')]}],
                    overwrite=False
                ).set_properties(**{'text-align': 'left'}).set_caption('Best Product Based on Sentiment Analysis'))

            # Visualization (optional): Bar chart of average polarities
            product_names = list(sentiment_results.keys())
            scores = [sentiment['average_polarity'] for sentiment in sentiment_results.values()]

            st.markdown('<h3 style="color: #1f77b4;">Average Polarity of Products</h3>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=product_names, y=scores, hue=product_names, palette="coolwarm", ax=ax, legend=False)
            ax.set_xlabel('Products', fontsize=14)
            ax.set_ylabel('Average Polarity', fontsize=14)
            ax.set_title('Average Polarity of Products', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

        else:
            st.write("No sentiment analysis results found.")

if __name__ == "__main__":
    main()



