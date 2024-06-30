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
    'A-Realme 12 5G_reviews.csv',
    'B-Poco F4 5G_reviews.csv',
    'C-Xiaomi 11 lite 128GB_reviews.csv',
    'D_Vivo v25 pro 5G_reviews.csv',
    'E-Oppo reno 11 5G - E_reviews.csv',
    'F-Redmi note 13pro 5G_reviews.csv',
    'G-Oneplus Nord 3- 5G_reviews.csv'
]

def fetch_product_name(df):
    """
    Fetches the product name from a DataFrame.
    Assumes the product name is in the first row, first column.
    """
    return df.iloc[0, 0]

dataframes = {}
for file in csv_files:
    try:
        df = pd.read_csv(file)
        product_name = fetch_product_name(df)
        dataframes[product_name] = df
    except FileNotFoundError:
        print(f"File not found: {file}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Function for sentiment analysis
def analyze_sentiment(review):
    """
    Analyzes sentiment of a given review using TextBlob.
    Returns polarity, subjectivity, and sentiment category (Positive, Negative, Neutral).
    """
    review = str(review)
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        sentiment_def = "Positive"
    elif polarity < 0:
        sentiment_def = "Negative"
    else:
        sentiment_def = "Neutral"
    return polarity, subjectivity, sentiment_def

def aggregate_sentiment_stats(reviews):
    """
    Aggregates sentiment statistics (average polarity, average subjectivity, overall sentiment counts).
    """
    polarities = []
    subjectivities = []
    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

    for review in reviews:
        polarity, subjectivity, sentiment_name = analyze_sentiment(review)
        polarities.append(polarity)
        subjectivities.append(subjectivity)
        sentiment_counts[sentiment_name] += 1

    avg_polarity = sum(polarities) / len(polarities) if polarities else 0
    avg_subjectivity = sum(subjectivities) / len(subjectivities) if subjectivities else 0
    overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)

    return avg_polarity, avg_subjectivity, overall_sentiment, sentiment_counts

def interpret_subjectivity_level(subjectivity):
    """
    Interprets subjectivity level into descriptive categories.
    """
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

def custom_review_tokenizer(text, vectorizer=None):
    """
    Tokenizes a review text using spaCy, filtering out stopwords and non-alphabetic tokens.
    Optionally filters tokens based on a TF-IDF vectorizer's vocabulary.
    """
    tokens = []
    doc = nlp(text)
    for token in doc:
        if token.is_alpha and not token.is_stop:
            if vectorizer and token.text.lower() in vectorizer.vocabulary_:
                tokens.append(token.text.lower())
            else:
                tokens.append('<UNK>')  # Replace OOV words with '<UNK>'
    return tokens

def compare_products_by_query(query):
    """
    Compares products based on similarity to a user query using TF-IDF cosine similarity.
    Returns sentiment analysis results for each product.
    """
    sentiments = {}
    vectorizer = TfidfVectorizer(stop_words='english')

    # Build vocabulary from all reviews
    all_reviews = [review for df in dataframes.values() for review in df['Review'].tolist()]
    vectorizer.fit(all_reviews)

    for product, df in dataframes.items():
        reviews = df['Review'].tolist()

        # Tokenize reviews with custom tokenizer
        reviews_tokens = [' '.join(custom_review_tokenizer(review, vectorizer)) for review in reviews]

        # Transform reviews into TF-IDF vectors
        reviews_transformed = vectorizer.transform(reviews_tokens)

        # Tokenize query with custom tokenizer
        query_tokens = ' '.join(custom_review_tokenizer(query, vectorizer))

        # Transform query into TF-IDF vector
        query_vec = vectorizer.transform([query_tokens])

        # Calculate cosine similarities between query and reviews
        cosine_similarities = cosine_similarity(query_vec, reviews_transformed).flatten()

        # Filter reviews based on cosine similarity
        threshold = 0.35  # Adjust threshold as needed
        filtered_reviews = [reviews[i] for i in range(len(reviews)) if cosine_similarities[i] > threshold]

        if filtered_reviews:
            try:
                avg_polarity, avg_subjectivity, overall_sentiment, sentiment_counts = aggregate_sentiment_stats(filtered_reviews)
                sentiments[product] = {
                    'average_polarity': avg_polarity,
                    'average_subjectivity': avg_subjectivity,
                    'overall_sentiment': overall_sentiment,
                    'Positive_Count': sentiment_counts["Positive"],
                    'Negative_Count': sentiment_counts["Negative"],
                    'Neutral_Count': sentiment_counts["Neutral"]
                }
            except Exception as e:
                st.error(f"Error processing {product}: {e}")

    return sentiments

def display_best_product(best_product):
    """
    Displays details of the best product based on sentiment analysis.
    """
    best_product_name = best_product[0]
    best_product_sentiment = best_product[1]

    best_product_data = {
        "Product": best_product_name,
        "Sentiment_Analysis": f'{best_product_sentiment["overall_sentiment"]}',
        "Subjectivity": f'{best_product_sentiment["average_subjectivity"]:.2f}',
        "Score": f'{best_product_sentiment["average_polarity"]:.2f}',
        "Positive_Counts": f'{best_product_sentiment["Positive_Count"]}',
        "Negative_Counts": f'{best_product_sentiment["Negative_Count"]}',
        "Neutral_Counts": f'{best_product_sentiment["Neutral_Count"]}'
    }
    if best_product_data:
        st.write(pd.DataFrame([best_product_data]).set_index('Product'))

def visualize_sentiment_pie_chart(size_value, sentiment_analysis):
    """
    Visualizes sentiment distribution of products using a pie chart.
    """
    labels = [product for product in sentiment_analysis.keys()]
    sizes = [data[size_value] for data in sentiment_analysis.values()]
    colors = plt.cm.Paired(range(len(labels)))

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax1.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=140,
                                       textprops={'fontsize': 10})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)

    legend = ax1.legend(loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), labels=labels)
    for text in legend.get_texts():
        text.set_color('black')
        text.set_fontsize(10)
    ax1.axis('equal')

    st.pyplot(fig1)

# Step 5: Streamlit App
def main():
    st.image("Mylogo.png", width=200)
    st.markdown('<div class="title">Product Sentiment Analysis</div>', unsafe_allow_html=True)

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

    st.markdown('<span style="color: #1f77b4; font-size: 1.2em;">Enter your query:</span>', unsafe_allow_html=True)

    user_query = st.text_input(' ', placeholder='Type your query here...', key="query")

    if user_query:
        query_sentiment = analyze_sentiment(user_query)
        query_polarity = query_sentiment[0]
        print("The Query Polarity", query_polarity)

        sentiment_analysis = compare_products_by_query(user_query)

        if sentiment_analysis:
            st.markdown('---')
            st.subheader('Sentiment Analysis Results')

            st.markdown('<h3 style="font-family:sans-serif; font-size:20px;">All Products Analysis</h3>', unsafe_allow_html=True)
            product_data = []
            for product, data in sentiment_analysis.items():
                product_data.append({
                    "Product": product,
                    "Sentiment_Analysis": data['overall_sentiment'],
                    "Subjectivity": f'{data["average_subjectivity"]:.2f}',
                    "Score": f'{data["average_polarity"]:.2f}',
                    "Positive_Counts": f'{data["Positive_Count"]}',
                    "Negative_Counts": f'{data["Negative_Count"]}',
                    "Neutral_Counts": f'{data["Neutral_Count"]}'
                })
            product_df = pd.DataFrame(product_data).set_index('Product')

            st.write(product_df)

            st.markdown('<h3 style="font-family:sans-serif; font-size:20px;">Product Recommendation</h3>', unsafe_allow_html=True)

            best_product_data = None

            if query_polarity >= 0:
                best_product = max(sentiment_analysis.items(), key=lambda x: x[1]['Positive_Count'])
                display_best_product(best_product)
                visualize_sentiment_pie_chart("Positive_Count", sentiment_analysis)
            else:
                best_product = max(sentiment_analysis.items(), key=lambda x: x[1]['Negative_Count'])
                display_best_product(best_product)
                visualize_sentiment_pie_chart("Negative_Count", sentiment_analysis)

            products = list(sentiment_analysis.keys())
            scores = [sentiment['average_polarity'] for sentiment in sentiment_analysis.values()]
            subjectivities = [sentiment['average_subjectivity'] for sentiment in sentiment_analysis.values()]

            st.markdown('<h3 style="font-family:sans-serif; font-size:20px;">Average Polarity of Products</h3>', unsafe_allow_html=True)
            product_names = list(sentiment_analysis.keys())
            scores = [sentiment['average_polarity'] for sentiment in sentiment_analysis.values()]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(product_names, scores, color=['skyblue', 'lightgreen', 'lightcoral', 'orange', 'lightpink', 'lightblue'])

            for bar in bars:
                bar.set_width(0.5)

            ax.set_xlabel('Products', fontsize=14)
            ax.set_ylabel('Average Polarity', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)

            ax.set_xticks(range(len(product_names)))
            ax.set_xticklabels(product_names, rotation=45, ha='right')

            st.pyplot(fig)

            subjectivity_data = {'Product': products, 'Average Subjectivity': subjectivities}
            subjectivity_df = pd.DataFrame(subjectivity_data).set_index('Product')

            subjectivity_df_sorted = subjectivity_df.sort_values(by='Average Subjectivity')

            st.markdown('<h3 style="font-family:sans-serif; font-size:20px;">Average Subjectivity per Product</h3>', unsafe_allow_html=True)
            st.table(subjectivity_df_sorted)

        else:
            st.write("No sentiment analysis results found.")

if __name__ == "__main__":
    main()
